/* ============================================================
   EdgeDistillDet Web UI - Training Management
   ============================================================ */

let trainingStartTime = null;
let wasRunning = false;
let eventSource = null;  // SSE connection for real-time logs
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
    const mode = typeof getCurrentTrainMode === 'function' ? getCurrentTrainMode() : 'full';
    const baseConfig = getConfigFromForm();
    const config = applyModeOverrides(baseConfig, mode);

    if (!config.distillation.student_weight) {
        showToast('请选择学生模型权重文件', 'warning');
        return;
    }
    if (!config.distillation.teacher_weight) {
        showToast('请选择教师模型权重文件', 'warning');
        return;
    }
    if (!config.training.data_yaml) {
        showToast('请填写数据集配置文件路径', 'warning');
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
        addLogLine(`[系统] 已启动训练（模式: ${mode}）`, 'info');
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
                if (typeof Animations !== 'undefined') {
                    Animations.TrainingButtonAnim.setState('success');
                    Animations.Celebration.trigger({ message: '训练完成!' });
                    Animations.ProgressBar.setStripesActive('progress-bar', false);
                }
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
        addLogLine('[系统] 训练已被用户停止', 'warning');
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
    if (AppState.logPollInterval) {
        clearInterval(AppState.logPollInterval);
        AppState.logPollInterval = null;
    }

    eventSource = new EventSource('/api/train/logs/stream');

    eventSource.onmessage = (evt) => {
        if (evt.data === '[DONE]') {
            disconnectSSELogStream();
            stopElapsedTimer();
            trainingStartTime = null;
            addLogLine('[系统] 日志流传输完成', 'success');
            // 触发训练完成庆祝动画
            if (typeof Animations !== 'undefined') {
                Animations.TrainingButtonAnim.setState('success');
                setTimeout(() => {
                    try {
                        Animations.Celebration.trigger({ message: '训练完成!' });
                    } catch (err) {
                        console.warn('庆祝动画触发失败，已忽略：', err);
                    }
                }, 300);
            }
            return;
        }

        try {
            const parsed = JSON.parse(evt.data);
            const line = typeof parsed === 'string' ? parsed : (parsed.line || evt.data);
            if (line) {
                const cleaned = sanitizeLogText(line);
                addLogLine(cleaned, classifyLogLine(cleaned), false);
                parseMetricsFromLogs([cleaned]);
                handleEpochProgress(cleaned);
                if (AppState.autoScroll) scrollToLogBottom();
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

    eventSource.onerror = () => {
        console.warn('[SSE] Connection error or closed.');
        AppState.useSSE = false;
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        managePolling();
    };

    console.log('[SSE] Real-time log stream connected.');
}

function disconnectSSELogStream() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    AppState.useSSE = false;
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

// ==================== Status Checking ====================
async function checkTrainingStatus() {
    try {
        const data = await API.get('/api/train/status');
        const running = !!data.running;
        wasRunning = running;

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

        data.logs.forEach((line) => {
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
    cleaned = cleaned.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, '');
    cleaned = cleaned.replace(/\u001b/g, '');
    cleaned = cleaned.replace(/\r/g, '');
    cleaned = cleaned.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, '');
    // Remove broken progress bar artifacts between '%' and next fraction like 128/128 or 1/8
    cleaned = cleaned.replace(/%\s*[^0-9A-Za-z\r\n]*?(?=\d+\/\d+)/g, '% ');
    // Remove trailing broken glyph sequences after a percent display
    cleaned = cleaned.replace(/%[^\r\n]*$/g, '%');
    return cleaned;
}

function normalizeLogoLine(text) {
    return String(text)
        .replace(/\u00A0/g, ' ')
        .replace(/\s+$/g, '');
}

function isAsciiLogoLine(text) {
    if (typeof text !== 'string' || text.length < 32) return false;
    const artChars = /[|_\\/\-]/;
    const repeatedSpaces = (text.match(/ {2,}/g) || []).length;
    const artCount = (text.match(/[|_\\/\-]/g) || []).length;
    return artChars.test(text) && repeatedSpaces >= 2 && artCount >= 3;
}

function addLogLine(text, type = 'info', autoScroll = true) {
    const container = document.getElementById('log-container');
    if (!container) return;

    const cleanedText = sanitizeLogText(text);
    const normalizedText = normalizeLogoLine(cleanedText);
    const isLogo = isAsciiLogoLine(normalizedText);

    if (!isLogo && currentLogoBlock.length > 0) {
        flushLogoBlock();
    }

    if (isLogo && currentLogoBlock.length > 0 && normalizedText === currentLogoBlock[0].text) {
        // Detected a new logo block starting immediately after the previous one.
        flushLogoBlock();
    }

    const line = document.createElement(isLogo ? 'pre' : 'div');
    line.className = `log-line ${type}${isLogo ? ' logo' : ''}`;
    line.textContent = cleanedText;
    container.appendChild(line);

    if (isLogo) {
        currentLogoBlock.push({ text: normalizedText, node: line });
    }

    while (container.children.length > 500) {
        container.removeChild(container.firstChild);
    }

    if (autoScroll && AppState.autoScroll) {
        scrollToLogBottom();
    }
    if (typeof adjustLogContainerHeight === 'function') {
        adjustLogContainerHeight();
    }
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

function classifyLogLine(line) {
    const lower = String(line).toLowerCase();
    if (lower.includes('error') || lower.includes('failed') || lower.includes('exception') || lower.includes('traceback')) return 'error';
    if (lower.includes('warning') || lower.includes('warn')) return 'warning';
    if (/\bepoch\b/i.test(line) || isTrainingEpochSummary(line)) return 'epoch';
    if (lower.includes('completed') || lower.includes('finished') || lower.includes('done')) return 'success';
    return 'info';
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
    const progress = extractEpochProgress(line);
    if (!progress || !progress.total) return;
    setEpochInfo(progress.current, progress.total);
}

function setEpochInfo(current, total) {
    const epochInfo = document.getElementById('epoch-info');
    if (epochInfo) {
        epochInfo.textContent = `Epoch: ${current} / ${total}`;
    }
    updateEpochProgress(current, total);
}

function clearLogs(showNotice = true) {
    const container = document.getElementById('log-container');
    if (!container) return;
    container.innerHTML = '';
    // Skip any existing historical logs until the server appends new ones.
    AppState.logOffset = Number.MAX_SAFE_INTEGER;
    lastLogoBlock = null;
    currentLogoBlock = [];

    if (showNotice) {
        addLogLine('[系统] 日志已清空', 'info');
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
