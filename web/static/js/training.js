/* ============================================================
   EdgeDistillDet Web UI - Training Management
   ============================================================ */

let trainingStartTime = null;
let wasRunning = false;
let eventSource = null;  // SSE connection for real-time logs
let sseReconnectAttempts = 0;
const MAX_SSE_RECONNECT = 5;
let lastLogoBlock = null;
let currentLogoBlock = [];

function getTrainingButtons() {
    return {
        start: document.getElementById('btn-start-training'),
        stop: document.getElementById('btn-stop-training')
    };
}

function setTrainingButtonsState(isRunning) {
    const { start, stop } = getTrainingButtons();
    if (start) {
        start.disabled = isRunning;
        start.classList.toggle('is-disabled', isRunning);
    }
    if (stop) {
        stop.disabled = !isRunning;
        stop.classList.toggle('is-disabled', !isRunning);
    }
}

function applyModeOverrides(config, mode) {
    return structuredClone(config);
}

// ==================== Training Control ====================
async function startTraining() {
    const mode = typeof getCurrentTrainMode === 'function' ? getCurrentTrainMode() : 'distill';
    const baseConfig = getConfigFromForm();
    const config = applyModeOverrides(baseConfig, mode);

    // 基础校验：所有模式都需要学生模型权重和数据集配置
    if (!config.distillation.student_weight) {
        showToast('请选择学生模型权重文件', 'warning');
        return;
    }
    if (!config.training.data_yaml) {
        showToast('请填写数据集配置文件路径', 'warning');
        return;
    }
    if (mode !== 'resume' && !config.distillation.teacher_weight) {
        showToast('请选择教师模型权重文件', 'warning');
        return;
    }

    const runName = config.output.name?.trim() || '';
    const projectPath = config.output.project?.trim() || 'runs/distill';
    const selectedResumeCandidate = AppState.selectedResumeCandidate;
    if (mode !== 'resume' || !selectedResumeCandidate) {
        if (runName) {
            const ok = await validateRunNameBeforeStart(projectPath, runName);
            if (!ok) {
                setTrainingModeSelectionEnabled(true);
                setTrainingButtonsState(false);
                return;
            }
        }
    }

    try {
        setTrainingButtonsState(true);
        setTrainingModeSelectionEnabled(false);
        // 触发按钮加载动画
        if (typeof Animations !== 'undefined') Animations.TrainingButtonAnim.setState('loading');

        // Save config first then start training.
        await API.post('/api/config/save', { name: 'distill_config.yaml', config });
        const requestBody = { config: 'distill_config.yaml', mode };
        if (mode === 'resume' && selectedResumeCandidate?.checkpoint) {
            requestBody.checkpoint = selectedResumeCandidate.checkpoint;
        }
        await API.post('/api/train/start', requestBody);

        trainingStartTime = Date.now();
        AppState.logOffset = 0;
        clearLogs(false);
        updateTrainUI('running');
        setEpochInfo(0, config.training.epochs || 0);
        startElapsedTimer();
        showToast('训练任务已启动', 'success');

        // 启用进度条条纹动画
        if (typeof Animations !== 'undefined') {
            Animations.ProgressBar.setStripesActive('progress-bar', true);
        }
        
        // 连接 WebSocket 实时通信（如果可用）
        if (typeof Animations !== 'undefined' && typeof Animations.RealTimeSocket !== 'undefined') {
            Animations.RealTimeSocket.on('epoch_end', (data) => {
                if (typeof updateEpochProgress === 'function') {
                    updateEpochProgress(data.epoch, data.total_epochs);
                    if (typeof Animations !== 'undefined') {
                        Animations.ProgressBar.update('progress-bar', data.epoch, data.total_epochs);
                        if (data.metrics) {
                            if (data.metrics.mAP50) Animations.CountUpAnim.animateStat('stat-map50', data.metrics.mAP50);
                            if (data.metrics.box_loss) Animations.CountUpAnim.animateStat('stat-loss', data.metrics.box_loss);
                        }
                    }
                }
            });
            Animations.RealTimeSocket.on('complete', () => {
                _wasPreviouslyRunning = false;  // 防止轮询重复触发
                _triggerTrainingComplete();
            });
        }

        // Connect SSE for real-time log streaming
        connectSSELogStream();

        managePolling();
    } catch (e) {
        setTrainingButtonsState(false);
        setTrainingModeSelectionEnabled(true);
        updateTrainUI('idle');
        if (typeof Animations !== 'undefined') Animations.TrainingButtonAnim.setState('error');
    }
}

async function stopTraining() {
    if (!wasRunning) {
        showToast('当前没有运行中的训练任务', 'info');
        return;
    }

    if (!confirm('确定要停止当前训练吗？')) return;

    try {
        await API.post('/api/train/stop');
        disconnectSSELogStream();
        stopElapsedTimer();
        setTrainingModeSelectionEnabled(true);
        // 清理动画状态
        if (typeof Animations !== 'undefined') {
            Animations.TrainingButtonAnim.setState('idle');
            Animations.ProgressBar.setStripesActive('progress-bar', false);
            if (typeof Animations.RealTimeSocket !== 'undefined') {
                Animations.RealTimeSocket.off('epoch_end');
                Animations.RealTimeSocket.off('distill_update');
                Animations.RealTimeSocket.off('complete');
            }
        }
        updateTrainUI('stopped');
        showToast('训练已停止', 'info');
        wasRunning = false;
    } catch (e) {
        // Error already handled in API helper.
    }
}

// ==================== SSE Real-time Log Stream ====================
function connectSSELogStream() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    AppState.useSSE = true;
    // 不清除轮询！改为混合模式：SSE 主导 + 轮询兜底

    eventSource = new EventSource('/api/train/logs/stream');

    let sseReceivedData = false;

    const handleLogStreamDone = () => {
        disconnectSSELogStream();
        trainingStartTime = null;
        wasRunning = false;
        _wasPreviouslyRunning = false;  // 防止轮询重复触发
        AppState.logOffset = Number.MAX_SAFE_INTEGER;
        // 触发训练完成庆祝动画
        _triggerTrainingComplete();
    };

    eventSource.onmessage = (evt) => {
        sseReceivedData = true;
        sseReconnectAttempts = 0; // 成功接收数据，重置重连计数

        if (evt.data === '[DONE]') {
            handleLogStreamDone();
            return;
        }

        try {
            const parsed = JSON.parse(evt.data);
            if (parsed.chunk) {
                appendLogChunk(parsed.chunk);
                return;
            }
            const line = typeof parsed === 'string' ? parsed : (parsed.line || evt.data);
            if (line) {
                const cleaned = sanitizeLogText(line);
                addLogLine(cleaned, classifyLogLine(cleaned), false);
                parseMetricsFromLogs([cleaned]);
                const epochDetail = handleEpochProgress(cleaned);
                if (epochDetail) {
                    addEpochDetailLine(epochDetail);
                }
                if (AppState.autoScroll) scrollToLogBottom();
                // SSE 已经提供了实时日志，轮询只是兜底，不需要调整 offset 的值。
            }
        } catch (e) {
            // Raw text fallback
            if (evt.data && !evt.data.startsWith(':')) {
                const cleaned = sanitizeLogText(evt.data);
                addLogLine(cleaned, 'info', false);
                if (AppState.autoScroll) scrollToLogBottom();
            }
        }
    };

    eventSource.addEventListener('done', () => {
        handleLogStreamDone();
    });

    eventSource.onerror = () => {
        console.warn(`[SSE] Connection error or closed. (attempt: ${sseReconnectAttempts})`);
        AppState.useSSE = false;
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        // 延迟重试 + 指数退避
        sseReconnectAttempts++;
        if (sseReconnectAttempts <= MAX_SSE_RECONNECT) {
            const delay = Math.min(1000 * Math.pow(2, sseReconnectAttempts), 10000);
            console.log(`[SSE] Reconnecting in ${delay}ms...`);
            setTimeout(() => {
                // 仅在训练仍在运行时重连
                if (wasRunning || (AppState.lastTrainingStatus && AppState.lastTrainingStatus.running)) {
                    connectSSELogStream();
                }
            }, delay);
        } else {
            console.log('[SSE] Max reconnect attempts reached, falling back to polling only.');
            managePolling();
        }
    };

    console.log('[SSE] Real-time log stream connected.');
}

function disconnectSSELogStream() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    AppState.useSSE = false;
    sseReconnectAttempts = 0;
    console.log('[SSE] Real-time log stream disconnected.');
}

function updateElapsedDisplay(statusData = null) {
    const elapsedTime = document.getElementById('elapsed-time');
    const expectedTime = document.getElementById('expected-time');
    const status = statusData || AppState.lastTrainingStatus || {};

    if (!elapsedTime || !trainingStartTime) {
        if (expectedTime) {
            expectedTime.textContent = '预计总耗时: --:--:--';
        }
        return;
    }

    const elapsed = Math.floor((Date.now() - trainingStartTime) / 1000);
    elapsedTime.textContent = `耗时: ${formatTime(Math.max(0, elapsed))}`;

    if (expectedTime) {
        if (status.total_epochs > 0) {
            if (status.current_epoch > 0) {
                const completed = Math.max(1, status.current_epoch);
                const estimateSeconds = Math.round((elapsed / completed) * status.total_epochs);
                expectedTime.textContent = `预计总耗时: ${formatTime(Math.max(0, estimateSeconds))}`;
            } else {
                expectedTime.textContent = '预计总耗时: 估算中...';
            }
        } else {
            expectedTime.textContent = '预计总耗时: --:--:--';
        }
    }
}

function startElapsedTimer() {
    if (AppState.elapsedIntervalId) {
        return;
    }

    updateElapsedDisplay();
    AppState.elapsedIntervalId = setInterval(() => updateElapsedDisplay(), 1000);
}

function stopElapsedTimer() {
    if (AppState.elapsedIntervalId) {
        clearInterval(AppState.elapsedIntervalId);
        AppState.elapsedIntervalId = null;
    }

    trainingStartTime = null;
    const expectedTime = document.getElementById('expected-time');
    if (expectedTime) {
        expectedTime.textContent = '预计总耗时: --:--:--';
    }
}

// ==================== Training Complete Animation ====================
function _triggerTrainingComplete() {
    stopElapsedTimer();
    setTrainingModeSelectionEnabled(true);
    if (typeof Animations !== 'undefined') {
        Animations.TrainingButtonAnim.setState('success');
        Animations.ProgressBar.setStripesActive('progress-bar', false);
        setTimeout(() => {
            try {
                Animations.Celebration.trigger({ message: '训练完成!' });
            } catch (err) {
                console.warn('庆祝动画触发失败，已忽略：', err);
            }
        }, 300);
    }
    updateTrainUI('completed');
}

// ==================== Status Checking ====================
let _wasPreviouslyRunning = false;  // 追踪上一次轮询的训练状态，用于检测"刚结束"

async function checkTrainingStatus() {
    try {
        const data = await API.get('/api/train/status');
        const running = !!data.running;
        wasRunning = running;

        // 检测训练"刚结束"的瞬间（running: true→false），触发完成动画
        if (_wasPreviouslyRunning && !running) {
            _wasPreviouslyRunning = false;
            // 延迟一点确保 SSE/DONE 优先处理（避免重复触发）
            setTimeout(() => {
                if (eventSource) return;  // SSE 还在连接中，由 SSE 路径处理
                _triggerTrainingComplete();
            }, 500);
        } else if (running) {
            _wasPreviouslyRunning = true;
        }

        if (data.start_time && !trainingStartTime) {
            trainingStartTime = Math.floor(data.start_time * 1000);
        }

        updateStatusBadge(running ? 'running' : 'idle');
        setTrainingButtonsState(running);

        const progress = document.getElementById('training-progress');
        const epochInfo = document.getElementById('epoch-info');
        const elapsedTime = document.getElementById('elapsed-time');

        if (progress) {
            progress.style.display = running ? 'block' : 'none';
        }

        if (typeof adjustLogContainerHeight === 'function') {
            adjustLogContainerHeight();
        }

        if (epochInfo && data.total_epochs > 0) {
            epochInfo.textContent = `Epoch: ${data.current_epoch} / ${data.total_epochs}`;
            updateEpochProgress(data.current_epoch, data.total_epochs);
        }

        if (data.mode) {
            updateTrainingModeStatus(data.mode);
        }

        setTrainingModeSelectionEnabled(!running);

        AppState.lastTrainingStatus = data;
        if (running) {
            startElapsedTimer();
        } else {
            stopElapsedTimer();
        }

        updateTrainUI(running ? 'running' : 'idle');

        // Auto-reconnect SSE if training is running but no SSE connection
        if (running && !eventSource) {
            connectSSELogStream();
        }
    } catch (e) {
        // Silent status failures avoid UI flicker.
    }
}

// ==================== Log Management ====================
function startLogPolling() {
    // Backward-compatible alias.
    managePolling();
}

async function fetchTrainingLogs() {
    if (AppState.useSSE) {
        return;
    }

    try {
        const offset = Number.isFinite(AppState.logOffset) ? AppState.logOffset : 0;
        const data = await API.get(`/api/train/logs?offset=${offset}&limit=120`);

        if (!Array.isArray(data.logs) || data.logs.length === 0) return;

        const seenLines = new Set(Array.from(document.querySelectorAll('#log-container .log-line'))
            .map(el => el.textContent?.trim())
            .filter(Boolean));

        data.logs.forEach((line) => {
            const cleaned = line.trim();
            if (!cleaned || seenLines.has(cleaned)) return;
            seenLines.add(cleaned);
            addLogLine(line, classifyLogLine(line), false);
            handleEpochProgress(line);
        });
        AppState.logOffset = data.offset + data.logs.length;

        if (AppState.autoScroll) scrollToLogBottom();
        parseMetricsFromLogs(data.logs);
    } catch (e) {
        // Silent polling failures keep page stable.
    }
}

function sanitizeLogText(text) {
    if (text == null) return '';
    let cleaned = String(text);
    // Strip ANSI escape sequences
    cleaned = cleaned.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, '');
    cleaned = cleaned.replace(/\u001b/g, '');
    cleaned = cleaned.replace(/\r/g, '');
    cleaned = cleaned.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, '');
    // Remove colored emoji and pictographs that can break ASCII banners
    cleaned = cleaned.replace(/\p{Extended_Pictographic}/gu, '');
    cleaned = cleaned.replace(/\uFE0F/g, '');
    cleaned = cleaned.replace(/\uFE0E/g, '');
    // Remove broken progress bar artifacts
    cleaned = cleaned.replace(/%\s*[^0-9A-Za-z\r\n]*?(?=\d+\/\d+)/g, '% ');
    cleaned = cleaned.replace(/%[^\r\n]*$/g, '%');
    return cleaned.replace(/\s+$/g, '');
}

function sanitizeLogChunk(text) {
    if (text == null) return '';
    let cleaned = String(text);
    cleaned = cleaned.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, '');
    cleaned = cleaned.replace(/\u001b/g, '');
    cleaned = cleaned.replace(/\uFE0F/g, '').replace(/\uFE0E/g, '');
    cleaned = cleaned.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, '');
    return cleaned;
}

function appendLogChunk(rawText) {
    if (rawText == null) return;
    const cleanedChunk = sanitizeLogChunk(rawText);
    if (!cleanedChunk) return;
    const container = document.getElementById('log-container');
    if (!container) return;

    const createStreamingLine = () => {
        const lineEl = document.createElement('div');
        lineEl.className = 'log-line info streaming';
        lineEl.dataset.logType = 'streaming';
        container.appendChild(lineEl);
        while (container.children.length > 800) container.removeChild(container.firstChild);
        return lineEl;
    };

    if (!AppState.currentStreamLine || !container.contains(AppState.currentStreamLine)) {
        AppState.currentStreamLine = createStreamingLine();
    }

    let currentLine = AppState.currentStreamLine;
    const tokens = cleanedChunk.split(/(\r\n|\r|\n)/);
    for (const token of tokens) {
        if (token === '\r' || token === '\n' || token === '\r\n') {
            if (token === '\r') {
                currentLine.textContent = '';
                continue;
            }
            const finishedText = currentLine.textContent;
            if (finishedText) {
                parseMetricsFromLogs([finishedText]);
                const epochDetail = handleEpochProgress(finishedText);
                if (epochDetail) addEpochDetailLine(epochDetail);
            }
            currentLine = createStreamingLine();
            AppState.currentStreamLine = currentLine;
            continue;
        }
        currentLine.textContent += token;
    }

    if (AppState.autoScroll) scrollToLogBottom();
}

/**
 * 智能日志分类器 — 返回 { type: string, display: boolean, priority: number }
 * type: 'system' | 'diagnostic' | 'model' | 'epoch' | 'progress' | 'metric' | 'success' | 'warning' | 'error' | 'info' | 'hidden' | 'raw-json'
 */
function classifyLogLine(line) {
    const text = String(line).trim();
    if (!text) return { type: 'hidden', display: false, priority: 0 };

    const lower = text.toLowerCase();

    // 1) 过滤 SSE 原始 JSON 行（浏览器控制台泄露）
    if (text.startsWith('{"line":') || text.match(/^\{.*"line"\s*:/)) {
        return { type: 'raw-json', display: false, priority: -1 };
    }

    // 2) 系统消息 [系统] / [INFO] / [DONE] / [INTERRUPT]
    if (/^\[系统\]|^\[DONE\]|^\[INTERRUPT\]|^\[FATAL\]/.test(text)) {
        return { type: 'system', display: true, priority: 90 };
    }
    if (/^\[(诊断|stderr|堆栈)\]/.test(text)) {
        return { type: 'diagnostic', display: true, priority: 5 };
    }
    if (lower.includes('training completed') || lower.includes('all results') || /训练完成|评估结果已保存/.test(text)) {
        return { type: 'success', display: true, priority: 95 };
    }

    // 3) 错误/警告
    if (/\b(error|exception|traceback|failed)\b/i.test(text) && !lower.includes('error-free')) {
        return { type: 'error', display: true, priority: 99 };
    }
    if (/\b(warning|warn)\b/i.test(text)) {
        return { type: 'warning', display: true, priority: 70 };
    }

    // 4) 模型结构信息（折叠显示，不隐藏）
    if ( /^\s+\d+\s+-1\s+\d+\s+ultralytics/i.test(text)
         || /^Model summary:/i.test(text)
         || /^Transferred \d+\/\d+ items/i.test(text)
         || /^Freezing layer/i.test(text)) {
        return { type: 'model', display: true, priority: 15 };
    }

    // 5) Epoch 训练进度条行 — 核心过滤逻辑
    // 匹配格式: "   3/20      2.98G      1.266      1.555 ... 38%"
    const progressMatch = text.match(/^\s*(\d+)\/(\d+)\s+[\d.]+G\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+:\s*(\d+)%/);
    if (progressMatch) {
        const pct = parseInt(progressMatch[3]);
        // 只显示 0% 和 100%（epoch 开始和结束），中间的进度更新隐藏
        if (pct === 0 || pct >= 99 || isNaN(pct)) {
            return { type: 'progress', display: true, priority: 30 };
        }
        return { type: 'progress', display: false, priority: 28 }; // 隐藏中间进度
    }

    // 6) Epoch 完成时的指标摘要
    // 格式: "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size"
    if (/^\s*Epoch\s+GPU_mem/.test(text)) {
        return { type: 'epoch-header', display: true, priority: 40 };
    }
    // Epoch 指标数据行（loss 值等）
    if (/^\s*\d+\/\d+\s+[\d.]+G?\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+$/.test(text) && !text.includes('%')) {
        return { type: 'metric', display: true, priority: 42 };
    }

    // 7) 验证指标行 (mAP 等)
    if (/(mAP50|mAP50-95|Box\(P)/i.test(text) && /Class\s+Images\s+Instances/.test(text)) {
        return { type: 'metric-header', display: true, priority: 45 };
    }
    // 类别级 mAP 数据行
    if (/\ball\b\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+/.test(text)) {
        return { type: 'metric', display: true, priority: 48 };
    }
    // 单类别 mAP 行
    if (/\w{4,}\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+/.test(text) && text.length < 80) {
        return { type: 'class-metric', display: true, priority: 46 };
    }

    // 8) 验证进度行 "Class ... 50%" — 隐藏中间进度
    if (/^                 Class\s+Images\s+Instances\s+Box\(P\s*:\s*\d+%$/i.test(text)) {
        return { type: 'val-progress', display: false, priority: 35 }; // 隐藏验证进度
    }

    // 9) 关键里程碑事件
    if (/starting training for \d+ epochs/i.test(text)
        || /\d+ epochs completed/i.test(text)
        || /validating .*\.pt/i.test(text)) {
        return { type: 'milestone', display: true, priority: 85 };
    }

    // 10) 环境信息（单行显示）
    if (/^(Ultralytics |AMP:|optimizer:|Image sizes )/i.test(text)) {
        return { type: 'env-info', display: true, priority: 25 };
    }
    if (/^(engine\\trainer:|train: |val: )/i.test(text)) {
        return { type: 'env-detail', display: false, priority: 10 }; // 隐藏详细参数
    }

    // 11) 分隔线
    if (/^={60,}$|^={10,}$/.test(text)) {
        return { type: 'separator', display: true, priority: 5 };
    }

    // 12) 更新提示
    if (/New https:\/\/pypi\.org/i.test(text)) {
        return { type: 'update-hint', display: false, priority: 3 }; // 隐藏更新提示
    }

    // 13) 默认显示
    return { type: 'info', display: true, priority: 20 };
}

function flushLogoBlock() {
    if (currentLogoBlock.length === 0) return;
    const currentTexts = currentLogoBlock.map(item => item.text);
    if (lastLogoBlock && arraysEqual(lastLogoBlock, currentTexts)) {
        currentLogoBlock.forEach(item => item.node.remove());
    } else {
        lastLogoBlock = currentTexts;
    }
    currentLogoBlock = [];
}

function arraysEqual(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function isTrainingEpochSummary(line) {
    const text = String(line);
    const epochMatch = text.match(/^\s*(\d+)\s*\/\s*(\d+)\b/);
    if (!epochMatch) return false;
    return /\bGPU_mem\b/i.test(text)
        || /\bbox_loss\b/i.test(text)
        || /\bcls_loss\b/i.test(text)
        || /\bdfl_loss\b/i.test(text)
        || /\bSize\b/i.test(text)
        || /\bInstances\b/i.test(text)
        || /\bit\/s\b/i.test(text)
        || /s</i.test(text);
}

function addLogLine(text, type = 'info', autoScroll = true) {
    const container = document.getElementById('log-container');
    if (!container) return;

    const cleanedText = sanitizeLogText(text);
    if (!cleanedText) return;

    // 使用智能分类器
    let classification = null;
    if (typeof classifyLogLine === 'function') {
        classification = classifyLogLine(cleanedText);
        if (type === 'error' || type === 'warning') {
            classification.type = type;
            classification.display = true;
            classification.priority = 99;
        }
    }

    // Force display when type is explicitly passed as a non-classifier string
    // (e.g., the initial "已启动训练" message with type='info')
    if (classification && (classification.type === 'raw-json' || classification.type === 'hidden')) return;

    const lineType = (classification && classification.type !== 'hidden' && classification.type !== 'raw-json')
        ? classification.type : type;

    // Logo detection: check if line looks like ASCII art (YOLO banner, etc.)
    const normalizedText = typeof normalizeLogoLine === 'function'
        ? normalizeLogoLine(cleanedText) : cleanedText;
    const isLogo = typeof isAsciiLogoLine === 'function'
        ? isAsciiLogoLine(normalizedText) : false;

    if (!isLogo && currentLogoBlock.length > 0) flushLogoBlock();
    if (isLogo && currentLogoBlock.length > 0 && normalizedText === currentLogoBlock[0].text) flushLogoBlock();

    const line = document.createElement(isLogo ? 'pre' : 'div');
    line.className = `log-line ${lineType}${isLogo ? ' logo' : ''}`;
    line.dataset.logType = lineType;

    const displayText = cleanedText;
    line.textContent = displayText;
    container.appendChild(line);

    if (isLogo) currentLogoBlock.push({ text: normalizedText, node: line });

    while (container.children.length > 800) container.removeChild(container.firstChild);

    if (autoScroll && AppState.autoScroll) scrollToLogBottom();
    if (typeof adjustLogContainerHeight === 'function') adjustLogContainerHeight();
}

function extractEpochProgress(line) {
    const text = String(line);
    const namedMatch = text.match(/\bEpoch\s*(\d+)\s*\/\s*(\d+)\b/i);
    if (namedMatch) {
        return {
            current: Number(namedMatch[1]),
            total: Number(namedMatch[2])
        };
    }
    const bareMatch = text.match(/^\s*(\d+)\s*\/\s*(\d+)\b/);
    if (bareMatch && isTrainingEpochSummary(text)) {
        return {
            current: Number(bareMatch[1]),
            total: Number(bareMatch[2])
        };
    }
    return null;
}

function handleEpochProgress(line) {
    // 优先解析结构化的 [BATCH_PROGRESS] 日志
    const batchMatch = line.match(/\[BATCH_PROGRESS\]\s+epoch=(\d+)\s+total=(\d+)\s+batch=(\d+)\/(\d+)\s+progress=([\d.]+)%\s+bar=\[([^\]]+)\]\s+batch_size=(\d+)\s+samples=([\d\.]+|unknown)\s+dataset_samples=([^\s]+)\s+phase=(\w+)/i);
    if (batchMatch) {
        const current = Number(batchMatch[1]);
        const total = Number(batchMatch[2]);
        const batchIndex = Number(batchMatch[3]);
        const batchTotal = Number(batchMatch[4]);
        const progress = batchMatch[5];
        const bar = batchMatch[6];
        const batchSize = Number(batchMatch[7]);
        const samples = batchMatch[8];
        const datasetSamples = batchMatch[9];
        const phase = batchMatch[10];

        if (total > 0) {
            const details = {
                current,
                total,
                loss: null,
                kdLoss: null,
                alpha: null,
                temp: null,
                batches: `${batchIndex}/${batchTotal} @ ${batchSize}`,
                batchSize,
                samples,
                datasetSamples: datasetSamples === 'unknown' ? '未知' : `${datasetSamples} images`,
                phase: phase.charAt(0).toUpperCase() + phase.slice(1),
                progress,
                bar
            };
            setEpochInfo(current, total, null, details);
            addEpochDetailLine(details);
            return details;
        }
    }

    // 优先解析结构化的 [EPOCH_PROGRESS] 日志
    const epMatch = line.match(/\[EPOCH_PROGRESS\]\s+epoch=(\d+)\s+total=(\d+)\s+loss=([\d.]+)\s+kd=([\d.]+)\s+alpha=([\d.]+)\s+temp=([\d.]+)\s+batches=([^\s]+)\s+batch_size=(\d+)\s+samples=([\d\.]+|unknown)\s+dataset_samples=([^\s]+)\s+phase=(\w+)/i);
    if (epMatch) {
        const current = Number(epMatch[1]);
        const total = Number(epMatch[2]);
        const loss = epMatch[3];
        const kdLoss = epMatch[4];
        const alpha = epMatch[5];
        const temp = epMatch[6];
        const batches = epMatch[7];
        const batchSize = epMatch[8];
        const samples = epMatch[9];
        const datasetSamples = epMatch[10];
        const phase = epMatch[11];

        if (total > 0) {
            const details = {
                current,
                total,
                loss,
                kdLoss,
                alpha,
                temp,
                batches: `${batches} @ ${batchSize}`,
                batchSize,
                samples,
                datasetSamples: datasetSamples === 'unknown' ? '未知' : `${datasetSamples} images`,
                phase: phase.charAt(0).toUpperCase() + phase.slice(1)
            };
            setEpochInfo(current, total, { loss, kdLoss, alpha, temp }, details);
            return details;
        }
    }

    // 兼容旧格式：ultralytics 原始 epoch 行（如 "100/100"）
    const progress = extractEpochProgress(line);
    if (!progress || !progress.total) return;
    setEpochInfo(progress.current, progress.total);
    return null;
}

function addEpochDetailLine(details) {
    if (!details) return;
    const container = document.getElementById('log-container');
    if (!container) return;

    const detailLine = document.createElement('div');
    detailLine.className = 'log-line epoch-detail';
    detailLine.dataset.logType = 'info';
    detailLine.textContent = `[Epoch ${details.current}/${details.total}] ${details.phase} | batches=${details.batches} | samples=${details.samples} | dataset=${details.datasetSamples} | loss=${details.loss} | kd=${details.kdLoss} | alpha=${details.alpha} | temp=${details.temp}`;
    container.appendChild(detailLine);
    while (container.children.length > 800) container.removeChild(container.firstChild);
    if (AppState.autoScroll) scrollToLogBottom();
}

function setEpochInfo(current, total, metrics = null, details = null) {
    const epochInfo = document.getElementById('epoch-info');
    if (epochInfo) {
        epochInfo.textContent = `Epoch: ${current} / ${total}`;
    }
    updateEpochProgress(current, total);
    setEpochDetails(details);

    // 更新蒸馏指标统计卡片（如果数据可用）
    if (metrics) {
        if (metrics.loss) updateStat('stat-loss', metrics.loss);
        if (metrics.kdLoss) updateStat('stat-kd-loss', metrics.kdLoss);
        if (metrics.alpha) updateStat('stat-alpha', metrics.alpha);
        if (metrics.temp)   updateStat('stat-temp', metrics.temp);
    }
}

function setEpochDetails(details) {
    const batchInfo = document.getElementById('batch-info');
    const sampleInfo = document.getElementById('sample-info');
    const datasetInfo = document.getElementById('dataset-info');
    const phaseInfo = document.getElementById('phase-info');

    if (batchInfo) {
        batchInfo.textContent = details?.batches ? `Batches: ${details.batches}` : 'Batches: --';
    }
    if (sampleInfo) {
        sampleInfo.textContent = details?.samples ? `Samples: ${details.samples}` : 'Samples: --';
    }
    if (datasetInfo) {
        datasetInfo.textContent = details?.datasetSamples ? `Dataset: ${details.datasetSamples}` : 'Dataset: --';
    }
    if (phaseInfo) {
        phaseInfo.textContent = details?.phase ? `Phase: ${details.phase}` : 'Phase: --';
    }
}

function clearLogs(showNotice = true) {
    const container = document.getElementById('log-container');
    if (!container) return;
    container.innerHTML = '';
    // When showNotice=false (called from startTraining), reset offset to 0
    // to capture new training logs from the fresh server-side log array.
    // When showNotice=true (manual clear), skip old logs by setting max offset.
    AppState.logOffset = showNotice ? Number.MAX_SAFE_INTEGER : 0;
    lastLogoBlock = null;
    currentLogoBlock = [];

    if (showNotice) {
        // Keep log container clear without adding synthetic notices.
    }
}

async function downloadLogs() {
    try {
        const response = await fetch('/api/train/logs/download');
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || '下载日志失败');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const fileName = `training_logs_${new Date().toISOString().slice(0,19).replace(/[:T]/g, '-')}.txt`;
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = fileName;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
        showToast('训练日志已下载', 'success');
    } catch (e) {
        showToast(e.message || '下载日志失败', 'error');
    }
}

function toggleAutoScroll() {
    AppState.autoScroll = !AppState.autoScroll;
    const btn = document.getElementById('autoscroll-btn');
    if (!btn) return;

    btn.classList.toggle('is-disabled', !AppState.autoScroll);
    btn.setAttribute('title', AppState.autoScroll ? '自动滚动：开' : '自动滚动：关');
}

function scrollToLogBottom() {
    const container = document.getElementById('log-container');
    if (!container) return;
    container.scrollTop = container.scrollHeight;
}

function adjustLogContainerHeight() {
    const leftSection = document.querySelector('.config-section');
    const logCard = document.querySelector('.log-card');
    const logContainer = document.getElementById('log-container');
    if (!leftSection || !logCard || !logContainer) return;

    const leftHeight = leftSection.getBoundingClientRect().height;
    const header = logCard.querySelector('.log-header');
    const progress = document.getElementById('training-progress');
    const toolbar = logCard.querySelector('.log-toolbar');

    let reservedSpace = 0;
    if (header) reservedSpace += header.offsetHeight;
    if (progress && progress.style.display !== 'none') reservedSpace += progress.offsetHeight;
    if (toolbar) reservedSpace += toolbar.offsetHeight;
    reservedSpace += 26; // 16px card padding + 10px gap buffer

    logCard.style.maxHeight = `${leftHeight}px`;
    logContainer.style.maxHeight = `${Math.max(120, leftHeight - reservedSpace)}px`;
}

window.addEventListener('resize', () => {
    if (typeof adjustLogContainerHeight === 'function') {
        adjustLogContainerHeight();
    }
});

// ==================== Metrics Parsing ====================
function parseMetricsFromLogs(logs) {
    const lastLogs = logs.slice(-16);

    for (const line of lastLogs) {
        // 解析结构化的 [BATCH_PROGRESS] 日志
        const batchMatch = line.match(/\[BATCH_PROGRESS\]\s+epoch=(\d+)\s+total=(\d+)\s+batch=(\d+)\/(\d+)\s+progress=([\d.]+)%\s+bar=\[([^\]]+)\]\s+batch_size=(\d+)\s+samples=([\d\.]+|unknown)\s+dataset_samples=([^\s]+)\s+phase=(\w+)/i);
        if (batchMatch) {
            const current = Number(batchMatch[1]);
            const total = Number(batchMatch[2]);
            const batchSize = Number(batchMatch[7]);
            const samples = batchMatch[8];
            const datasetSamples = batchMatch[9];
            const phase = batchMatch[10];
            setEpochInfo(current, total, null, {
                batches: `${batchMatch[3]}/${batchMatch[4]} @ ${batchSize}`,
                samples,
                datasetSamples: datasetSamples === 'unknown' ? '未知' : `${datasetSamples} images`,
                phase: phase.charAt(0).toUpperCase() + phase.slice(1)
            });
            continue;
        }

        // 解析结构化的 EPOCH_PROGRESS 行
        const epMatch = line.match(/\[EPOCH_PROGRESS\].*?epoch=(\d+)\s+total=(\d+)\s+loss=([\d.]+)\s+kd=([\d.]+)\s+alpha=([\d.]+)\s+temp=([\d.]+)\s+batches=([^\s]+)\s+batch_size=(\d+)\s+samples=([\d\.]+|unknown)\s+dataset_samples=([^\s]+)\s+phase=(\w+)/i);
        if (epMatch) {
            updateStat('stat-loss', epMatch[3]);
            updateStat('stat-kd-loss', epMatch[4]);
            updateStat('stat-alpha', epMatch[5]);
            updateStat('stat-temp', epMatch[6]);
            setEpochInfo(Number(epMatch[1]), Number(epMatch[2]), null, {
                batches: `${epMatch[7]} @ ${epMatch[8]}`,
                samples: epMatch[9],
                datasetSamples: epMatch[10] === 'unknown' ? '未知' : `${epMatch[10]} images`,
                phase: epMatch[11].charAt(0).toUpperCase() + epMatch[11].slice(1)
            });
        }

        // 解析 TRAIN_INFO 数据集信息
        const infoMatch = line.match(/\[TRAIN_INFO\].*?images=(\d+)\s*classes=(\d+)/i);
        if (infoMatch) {
            const elImages = document.getElementById('stat-images');
            if (elImages) elImages.textContent = infoMatch[1];
            const elClasses = document.getElementById('stat-classes');
            if (elClasses) elClasses.textContent = infoMatch[2];
        }

        // 兼容旧格式：ultralytics 原始 loss/mAP 输出
        const lossMatch = line.match(/box_loss=([\d.]+)\s+cls_loss=([\d.]+)\s+dfl_loss=([\d.]+)/i);
        if (lossMatch) {
            updateStat('stat-loss', (+lossMatch[1] + +lossMatch[2] + +lossMatch[3]).toFixed(4));
        }

        const mapMatch = line.match(/mAP50=([\d.]+)\s+mAP50-95=([\d.]+)/i);
        if (mapMatch) {
            updateStat('stat-map50', mapMatch[1]);
            updateStat('stat-map95', mapMatch[2]);
        }

        const lrMatch = line.match(/lr0=([\d.eE-]+)/i);
        if (lrMatch) {
            updateStat('stat-lr', lrMatch[1]);
        }

        const epochMatch = line.match(/Epoch\s*(\d+)\s*\/\s*(\d+)/i);
        if (epochMatch) {
            const current = Number(epochMatch[1]);
            const total = Number(epochMatch[2]);
            const epochInfo = document.getElementById('epoch-info');
            if (epochInfo) epochInfo.textContent = `Epoch: ${current} / ${total}`;
            updateEpochProgress(current, total);
        }
    }

    const hasData = document.getElementById('stat-loss')?.textContent !== '--';
    const quickStats = document.getElementById('quick-stats');
    if (quickStats) {
        quickStats.style.display = hasData ? 'grid' : 'none';
    }
}

function updateEpochProgress(current, total) {
    const bar = document.getElementById('progress-bar');
    if (!bar || !total) return;
    const percent = Math.max(0, Math.min(100, (current / total) * 100));
    bar.style.width = `${percent.toFixed(1)}%`;
}

function updateStat(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el) return;

    // 使用 CountUp 数字滚动动画
    if (typeof Animations !== 'undefined' && !Animations.prefersReducedMotion) {
        const numValue = parseFloat(value);
        if (!isNaN(numValue)) {
            // 判断是上升还是下降来决定高亮颜色
            const prevVal = parseFloat(el.dataset.currentValue || el.textContent) || 0;
            el.classList.remove('highlight-up', 'highlight-down');
            void el.offsetWidth; // 触发 reflow
            el.classList.add(numValue > prevVal && elementId.includes('map') ? 'highlight-up' :
                             numValue < prevVal && !elementId.includes('map') ? 'highlight-up' : '');
            Animations.CountUpAnim.animateStat(elementId, value);
            return;
        }
    }
    el.textContent = value;
}

// ==================== UI Updates ====================
function updateTrainUI(status) {
    const badge = document.getElementById('train-status-badge');
    const progress = document.getElementById('training-progress');
    const connStatus = document.getElementById('connection-status');
    const isRunning = status === 'running';

    if (badge) {
        badge.className = `badge ${status}`;
        badge.textContent = {
            idle: '空闲',
            running: '训练中',
            stopped: '已停止',
            completed: '已完成'
        }[status] || status;
    }

    if (progress) {
        progress.style.display = isRunning ? 'block' : 'none';
    }

    setTrainingButtonsState(isRunning);

    if (connStatus) {
        connStatus.className = `status-indicator ${isRunning ? 'training' : 'online'}`;
        const label = connStatus.querySelector('span:last-child');
        if (label) label.textContent = isRunning ? '训练中' : '服务正常';
    }
}

function updateStatusBadge(status) {
    const badge = document.getElementById('train-status-badge');
    if (!badge) return;

    badge.className = `badge ${status}`;
    badge.textContent = status === 'running' ? '训练中' : '空闲';
}

// ==================== Utilities ====================
function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// ==================== Logo Detection (local fallback) ====================
// These functions are required by addLogLine() but were previously missing,
// causing ReferenceError that prevented ALL log lines from displaying.
function normalizeLogoLine(text) {
    if (text == null || typeof text !== 'string') return '';
    return text.replace(/\s+$/g, '');
}

function isAsciiLogoLine(text) {
    if (!text || text.length < 4) return false;
    let nonAlphaCount = 0;
    for (let i = 0; i < text.length; i++) {
        const c = text.charCodeAt(i);
        if ((c >= 33 && c <= 126) && !/[a-zA-Z0-9\s]/.test(text[i])) {
            nonAlphaCount++;
        }
    }
    return nonAlphaCount > text.length * 0.35;
}
