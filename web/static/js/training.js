/* ============================================================
   EdgeDistillDet Web UI - Training Management
   ============================================================ */

let trainingStartTime = null;
let wasRunning = false;
let eventSource = null;  // SSE connection for real-time logs

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
    const cloned = structuredClone(config);
    if (mode === 'fast') {
        cloned.training.epochs = Math.max(3, Math.min(cloned.training.epochs, 20));
        cloned.training.close_mosaic = Math.min(cloned.training.close_mosaic || 20, 8);
    }
    return cloned;
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

    try {
        setTrainingButtonsState(true);
        // 触发按钮加载动画
        if (typeof Animations !== 'undefined') Animations.TrainingButtonAnim.setState('loading');

        // Save config first then start training.
        await API.post('/api/config/save', { name: 'distill_config.yaml', config });
        await API.post('/api/train/start', {
            config: 'distill_config.yaml',
            mode
        });

        trainingStartTime = Date.now();
        AppState.logOffset = 0;
        clearLogs(false);
        addLogLine(`[系统] 已启动训练（模式: ${mode}）`, 'info');
        updateTrainUI('running');
        showToast('训练任务已启动', 'success');

        // 启动蒸馏流动画
        if (typeof Animations !== 'undefined') Animations.DistillFlowCanvas.start();
        
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
            Animations.RealTimeSocket.on('distill_update', (data) => {
                if (typeof Animations !== 'undefined' && data.alpha !== undefined) {
                    Animations.DistillFlowCanvas.setAlpha(data.alpha);
                    Animations.DistillFlowCanvas.setKDLoss(data.kd_loss || 3.0);
                }
            });
            Animations.RealTimeSocket.on('complete', () => {
                if (typeof Animations !== 'undefined') {
                    Animations.TrainingButtonAnim.setState('success');
                    Animations.Celebration.trigger({ message: '训练完成!' });
                    Animations.DistillFlowCanvas.stop();
                    Animations.ProgressBar.setStripesActive('progress-bar', false);
                }
            });
        }

        // Connect SSE for real-time log streaming
        connectSSELogStream();

        managePolling();
    } catch (e) {
        setTrainingButtonsState(false);
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
        // 清理动画状态
        if (typeof Animations !== 'undefined') {
            Animations.TrainingButtonAnim.setState('idle');
            Animations.DistillFlowCanvas.stop();
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
    }

    eventSource = new EventSource('/api/train/logs/stream');

    eventSource.onmessage = (evt) => {
        if (evt.data === '[DONE]') {
            disconnectSSELogStream();
            addLogLine('[系统] 日志流传输完成', 'success');
            // 触发训练完成庆祝动画
            if (typeof Animations !== 'undefined') {
                Animations.TrainingButtonAnim.setState('success');
                setTimeout(() => {
                    Animations.Celebration.trigger({ message: '训练完成!' });
                }, 300);
                // 停止蒸馏流动画
                Animations.DistillFlowCanvas.stop();
            }
            return;
        }

        try {
            const parsed = JSON.parse(evt.data);
            const line = typeof parsed === 'string' ? parsed : (parsed.line || evt.data);
            if (line) {
                addLogLine(line, classifyLogLine(line), false);
                parseMetricsFromLogs([line]);
                if (AppState.autoScroll) scrollToLogBottom();
            }
        } catch (e) {
            // Raw text fallback
            if (evt.data && !evt.data.startsWith(':')) {
                addLogLine(evt.data, 'info', false);
                if (AppState.autoScroll) scrollToLogBottom();
            }
        }
    };

    eventSource.onerror = () => {
        console.warn('[SSE] Connection error or closed.');
        // Don't auto-reconnect here — checkTrainingStatus will handle it.
    };

    console.log('[SSE] Real-time log stream connected.');
}

function disconnectSSELogStream() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
        console.log('[SSE] Real-time log stream disconnected.');
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

        if (epochInfo && data.total_epochs > 0) {
            epochInfo.textContent = `Epoch: ${data.current_epoch} / ${data.total_epochs}`;
            updateEpochProgress(data.current_epoch, data.total_epochs);
        }

        if (elapsedTime && trainingStartTime) {
            const elapsed = Math.floor((Date.now() - trainingStartTime) / 1000);
            elapsedTime.textContent = `耗时: ${formatTime(Math.max(0, elapsed))}`;
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
    try {
        const offset = Number.isFinite(AppState.logOffset) ? AppState.logOffset : 0;
        const data = await API.get(`/api/train/logs?offset=${offset}&limit=120`);

        if (!Array.isArray(data.logs) || data.logs.length === 0) return;

        data.logs.forEach((line) => addLogLine(line, classifyLogLine(line), false));
        AppState.logOffset = data.offset + data.logs.length;

        if (AppState.autoScroll) scrollToLogBottom();
        parseMetricsFromLogs(data.logs);
    } catch (e) {
        // Silent polling failures keep page stable.
    }
}

function addLogLine(text, type = 'info', autoScroll = true) {
    const container = document.getElementById('log-container');
    if (!container) return;

    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.textContent = text;
    container.appendChild(line);

    while (container.children.length > 500) {
        container.removeChild(container.firstChild);
    }

    if (autoScroll && AppState.autoScroll) {
        scrollToLogBottom();
    }
}

function classifyLogLine(line) {
    const lower = String(line).toLowerCase();
    if (lower.includes('error') || lower.includes('failed') || lower.includes('exception') || lower.includes('traceback')) return 'error';
    if (lower.includes('warning') || lower.includes('warn')) return 'warning';
    if (lower.includes('epoch') || /epoch\s*\d+/i.test(line)) return 'epoch';
    if (lower.includes('completed') || lower.includes('finished') || lower.includes('done')) return 'success';
    return 'info';
}

function clearLogs(showNotice = true) {
    const container = document.getElementById('log-container');
    if (!container) return;
    container.innerHTML = '';
    AppState.logOffset = 0;

    if (showNotice) {
        addLogLine('[系统] 日志已清空', 'info');
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
