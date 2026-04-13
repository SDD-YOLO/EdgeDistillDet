/* ============================================================
   EdgeDistillDet Web UI - Main Application Logic
   ============================================================ */

// Global State
const AppState = {
    currentTab: 'training',
    autoScroll: true,
    logPollInterval: null,
    statusPollInterval: null,
    logOffset: 0,
    metricsInitialized: false,
    filePickerTarget: null,      // 当前文件选择器目标输入ID
    filePickerAccept: null       // 当前文件选择器accept属性
};

// ==================== API Helper ====================
const API = {
    async request(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: { 'Content-Type': 'application/json', ...options.headers },
                ...options
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '请求失败');
            return data;
        } catch (error) {
            console.error('API Error:', error);
            showToast(error.message, 'error');
            throw error;
        }
    },

    get(url) { return this.request(url); },
    post(url, body) { return this.request(url, { method: 'POST', body: JSON.stringify(body) }); }
};

// ==================== Toast Notifications ====================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const icons = {
        success: 'check_circle',
        error: 'error',
        warning: 'warning',
        info: 'info'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="material-icons">${icons[type]}</span>
        <span>${message}</span>
    `;
    container.appendChild(toast);
    
    setTimeout(() => toast.remove(), 3000);
}

// ==================== Tab Navigation ====================

function initTabs() {
    const tabs = document.getElementById('main-tabs');
    if (!tabs) return;

    // Tab switching is handled by onclick in HTML, but we add keyboard navigation
    tabs.addEventListener('keydown', (e) => {
        const tabItems = Array.from(tabs.querySelectorAll('.tab-btn'));
        const currentIndex = tabItems.findIndex(t => t === document.activeElement);
        
        if (e.key === 'ArrowRight' && currentIndex < tabItems.length - 1) {
            tabItems[currentIndex + 1].focus();
        } else if (e.key === 'ArrowLeft' && currentIndex > 0) {
            tabItems[currentIndex - 1].focus();
        } else if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            switchTab(document.activeElement.dataset.tab);
        }
    });
}

function initFilePicker() {
    const hiddenInput = document.getElementById('hidden-file-input');
    if (hiddenInput) {
        hiddenInput.addEventListener('change', function() {
            if (AppState.filePickerTarget) {
                handleFileSelect(AppState.filePickerTarget);
            }
        });
    }
}

function switchTab(tabValue) {
    AppState.currentTab = tabValue;
    
    // Update tabs visual state (using .tab-btn[data-tab] instead of md-tab)
    document.querySelectorAll('#main-tabs .tab-btn').forEach(btn => {
        const isActive = btn.dataset.tab === tabValue;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    // Show/hide panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `panel-${tabValue}`);
    });

    // Tab-specific initialization
    if (tabValue === 'metrics' && typeof initCharts === 'function' && !AppState.metricsInitialized) {
        setTimeout(() => {
            initCharts();
            AppState.metricsInitialized = true;
        }, 120);
    }

    // Start/stop polling based on active tab
    managePolling();
}

function managePolling() {
    // Clear existing intervals
    if (AppState.logPollInterval) clearInterval(AppState.logPollInterval);
    if (AppState.statusPollInterval) clearInterval(AppState.statusPollInterval);

    // Always poll status when training might be running
    AppState.statusPollInterval = setInterval(() => checkTrainingStatus(), 2000);
    checkTrainingStatus();

    // Poll logs more frequently on training tab
    if (AppState.currentTab === 'training') {
        AppState.logPollInterval = setInterval(() => fetchTrainingLogs(), 1200);
        fetchTrainingLogs();
    }
}

// ==================== Slider Helpers ====================
function initSliders() {
    document.querySelectorAll('.md-slider').forEach(slider => {
        updateSliderValue(slider);
    });
}

function updateSliderValue(slider) {
    const valEl = document.getElementById(`${slider.id}-val`);
    if (valEl) {
        valEl.textContent = parseFloat(slider.value).toFixed(2);
    }
}

function editSliderValue(slider, valueEl) {
    // Create input field for direct editing
    const currentValue = parseFloat(slider.value);
    const min = parseFloat(slider.min) || 0;
    const max = parseFloat(slider.max) || 100;

    const inputEl = document.createElement('input');
    inputEl.type = 'number';
    inputEl.value = currentValue.toFixed(2);
    inputEl.min = min;
    inputEl.max = max;
    inputEl.className = 'slider-value-input';
    inputEl.step = 'any';

    // Get computed styles from the span to match exactly
    const spanComputedStyle = window.getComputedStyle(valueEl);
    const spanRect = valueEl.getBoundingClientRect();
    const spanPaddingTop = spanComputedStyle.paddingTop;
    const spanPaddingBottom = spanComputedStyle.paddingBottom;
    const spanPaddingLeft = spanComputedStyle.paddingLeft;
    const spanPaddingRight = spanComputedStyle.paddingRight;
    const spanBorderRadius = spanComputedStyle.borderRadius;

    // Use fixed sizing to prevent any layout shift
    inputEl.style.cssText = `
        width: ${spanRect.width}px;
        height: ${spanRect.height}px;
        padding: ${spanPaddingTop} ${spanPaddingRight} ${spanPaddingBottom} ${spanPaddingLeft};
        border: 1px solid var(--md-primary);
        border-radius: ${spanBorderRadius};
        background: var(--md-surface);
        color: var(--md-on-surface);
        font-size: 12px;
        line-height: 1;
        text-align: center;
        outline: none;
        box-sizing: border-box;
        margin: 0;
        vertical-align: top;
        flex-shrink: 0;
    `;

    // Replace the span with input in the same container - use visibility to preserve layout
    const parent = valueEl.parentNode;
    // Freeze parent layout during swap to prevent jitter
    parent.style.gridAutoFlow = 'column';
    valueEl.style.visibility = 'hidden';
    valueEl.style.position = 'absolute';
    valueEl.style.pointerEvents = 'none';

    parent.insertBefore(inputEl, valueEl);
    inputEl.focus();
    inputEl.select();

    const finishEdit = () => {
        let newValue = parseFloat(inputEl.value);

        // Validate and clamp value
        if (isNaN(newValue)) {
            newValue = currentValue;
        } else {
            newValue = Math.max(min, Math.min(max, newValue));
        }

        // Update slider and display value with 2 decimal places
        slider.value = newValue;
        updateSliderValue(slider);
        slider.dispatchEvent(new Event('input', { bubbles: true }));

        // Restore span - remove input first, then restore span visually
        inputEl.remove();
        valueEl.style.position = '';
        valueEl.style.visibility = '';
        valueEl.style.pointerEvents = '';
        parent.style.gridAutoFlow = '';
    };

    inputEl.addEventListener('blur', finishEdit);
    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            finishEdit();
        } else if (e.key === 'Escape') {
            inputEl.remove();
            valueEl.style.position = '';
            valueEl.style.visibility = '';
            valueEl.style.pointerEvents = '';
            parent.style.gridAutoFlow = '';
        }
    });
}

// ==================== File Browser (Native Picker) ====================

/**
 * 触发隐藏的文件选择器
 * @param {string} targetId - 目标文本输入框ID
 * @param {string} accept - 文件类型限制，如 '.pt,.pth'
 */
function triggerFilePicker(targetId, accept = '') {
    AppState.filePickerTarget = targetId;
    AppState.filePickerAccept = accept;
    const hiddenInput = document.getElementById('hidden-file-input');
    if (hiddenInput) {
        hiddenInput.accept = accept;
        hiddenInput.click(); // 触发文件选择对话框
    }
}

/**
 * Native file picker handler - reads selected file and populates the target input
 * 现在支持两种调用方式：
 * 1. handleFileSelect(fileInputElement, targetId) - 传统方式
 * 2. handleFileSelect(targetId) - 使用全局隐藏输入和AppState.filePickerTarget
 * @param {HTMLInputElement|string} fileInputOrTargetId - 文件输入元素或目标ID
 * @param {string} targetId - 可选，目标输入框ID（如果第一个参数是fileInputElement）
 */
function handleFileSelect(fileInputOrTargetId, targetId) {
    let fileInput, actualTargetId;
    
    // 判断调用方式
    if (typeof fileInputOrTargetId === 'string') {
        // 新方式: handleFileSelect(targetId)
        actualTargetId = fileInputOrTargetId;
        fileInput = document.getElementById('hidden-file-input');
    } else {
        // 传统方式: handleFileSelect(fileInput, targetId)
        fileInput = fileInputOrTargetId;
        actualTargetId = targetId;
    }
    
    if (!fileInput || !actualTargetId) return;
    
    if (fileInput.files && fileInput.files.length > 0) {
        const file = fileInput.files[0];
        // Use webkitRelativePath for full path if available, otherwise use name
        const filePath = file.webkitRelativePath || file.name;
        const targetEl = document.getElementById(actualTargetId);
        if (targetEl) {
            targetEl.value = filePath;
            // Visual feedback - briefly highlight the input
            targetEl.style.borderColor = 'var(--md-success)';
            setTimeout(() => { targetEl.style.borderColor = ''; }, 1500);
        }
        showToast(`已选择文件: ${file.name}`, 'success');
        
        // 重置隐藏输入以便可以选择相同文件再次触发change事件
        fileInput.value = '';
        // 清除状态
        AppState.filePickerTarget = null;
        AppState.filePickerAccept = null;
    }
}

// Compatibility stubs: legacy file dialog has been removed in favor of native picker.
function browseFile(targetId) {
    triggerFilePicker(targetId);
}

function closeFileDialog() {}

function confirmFileSelection() {}

// ==================== Config Management ====================
function getConfigFromForm() {
    return {
        distillation: {
            student_weight: document.getElementById('student-weight').value,
            teacher_weight: document.getElementById('teacher-weight').value,
            alpha_init: parseFloat(document.getElementById('alpha-init').value),
            T_max: parseFloat(document.getElementById('t-max').value),
            T_min: parseFloat(document.getElementById('t-min').value),
            warm_epochs: parseInt(document.getElementById('warm-epochs').value) || 5,
            w_kd: parseFloat(document.getElementById('w-kd').value),
            w_focal: parseFloat(document.getElementById('w-focal').value),
            w_feat: parseFloat(document.getElementById('w-feat').value),
            scale_boost: parseFloat(document.getElementById('scale-boost').value),
            focal_gamma: parseFloat(document.getElementById('focal-gamma').value)
        },
        training: {
            data_yaml: document.getElementById('data-yaml').value,
            device: document.getElementById('device').value,
            epochs: parseInt(document.getElementById('epochs').value) || 150,
            imgsz: parseInt(document.getElementById('imgsz').value) || 640,
            batch: parseInt(document.getElementById('batch-size').value) || -1,
            workers: parseInt(document.getElementById('workers').value) || 8,
            lr0: parseFloat(document.getElementById('lr0').value),
            lrf: parseFloat(document.getElementById('lrf').value),
            warmup_epochs: parseFloat(document.getElementById('warmup-lr-epochs').value),
            mosaic: parseFloat(document.getElementById('mosaic').value),
            mixup: parseFloat(document.getElementById('mixup').value),
            close_mosaic: parseInt(document.getElementById('close-mosaic').value) || 20,
            amp: document.getElementById('amp').checked
        },
        output: {
            project: document.getElementById('project').value,
            name: document.getElementById('run-name').value
        }
    };
}

function populateForm(config) {
    const d = config.distillation || {};
    const t = config.training || {};
    const o = config.output || {};

    // Distillation params
    setVal('student-weight', d.student_weight);
    setVal('teacher-weight', d.teacher_weight);
    setVal('alpha-init', d.alpha_init, 0.5);
    setVal('t-max', d.T_max, 6);
    setVal('t-min', d.T_min, 1.5);
    setVal('warm-epochs', d.warm_epochs, 5);
    setVal('w-kd', d.w_kd, 0.5);
    setVal('w-focal', d.w_focal, 0.3);
    setVal('w-feat', d.w_feat, 0);
    setVal('scale-boost', d.scale_boost, 2);
    setVal('focal-gamma', d.focal_gamma, 2);

    // Training params
    setVal('data-yaml', t.data_yaml);
    setVal('device', t.device, '0');
    setVal('epochs', t.epochs, 150);
    setVal('imgsz', t.imgsz, 640);
    setVal('batch-size', t.batch, -1);
    setVal('workers', t.workers, 8);
    setVal('lr0', t.lr0, 0.01);
    setVal('lrf', t.lrf, 0.1);
    setVal('warmup-lr-epochs', t.warmup_epochs, 3);
    setVal('mosaic', t.mosaic, 0.8);
    setVal('mixup', t.mixup, 0.1);
    setVal('close-mosaic', t.close_mosaic, 20);
    setVal('amp', t.amp !== undefined ? t.amp : true, true, true);

    // Output
    setVal('project', o.project, 'runs/distill');
    setVal('run-name', o.name, 'adaptive_kd_v1');

    // Update slider displays
    document.querySelectorAll('.md-slider').forEach(updateSliderValue);
}

function setVal(id, value, defaultValue, isCheckbox = false) {
    const el = document.getElementById(id);
    if (!el) return;
    if (isCheckbox) {
        el.checked = value !== undefined ? value : defaultValue;
    } else {
        const resolved = value !== undefined && value !== null
            ? value
            : (defaultValue !== undefined ? defaultValue : '');
        el.value = resolved;
        // Trigger events for enhanced UI components to update their visual state
        if (el.type === 'range' || el.classList.contains('md-slider')) {
            // For enhanced sliders
            el.dispatchEvent(new Event('input', { bubbles: true }));
        } else if (el.type === 'number') {
            // For enhanced numeric inputs
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }
}

async function loadDefaultConfig() {
    try {
        const data = await API.get('/api/config/distill_config.yaml');
        populateForm(data.config);
    } catch (e) {
        console.log('使用默认配置');
    }
}

async function loadConfig() {
    try {
        const data = await API.get('/api/configs');
        const configs = Object.keys(data.configs);
        if (configs.length === 0) {
            showToast('未找到配置文件', 'warning');
            return;
        }
        // Load first available config (distill_config.yaml preferred)
        const target = configs.includes('distill_config.yaml') ? 'distill_config.yaml' : configs[0];
        const result = await API.get(`/api/config/${target}`);
        populateForm(result.config);
        showToast(`已加载配置: ${target}`, 'success');
    } catch (e) {
        showToast('加载配置失败', 'error');
    }
}

async function saveConfig() {
    try {
        const config = getConfigFromForm();
        const name = 'distill_config.yaml';
        await API.post('/api/config/save', { name, config });
        showToast('配置已保存', 'success');
    } catch (e) {
        showToast('保存配置失败', 'error');
    }
}

function resetForm() {
    if (confirm('确定要重置所有参数到默认值吗?')) {
        loadDefaultConfig();
        showToast('表单已重置', 'info');
    }
}

// ==================== Training Mode Selection Enhancement ====================
function initTrainingModeSelection() {
    const modeOptions = document.querySelectorAll('.mode-option');
    if (!modeOptions.length) return;
    
    // Keep radio/label state deterministic and avoid transient transform flicker.
    modeOptions.forEach(option => {
        const radioInput = option.querySelector('input[type="radio"]');
        if (!radioInput) return;

        option.addEventListener('click', function() {
            radioInput.checked = true;
            radioInput.dispatchEvent(new Event('change', { bubbles: true }));
        });
        
        radioInput.addEventListener('change', function() {
            if (this.checked) {
                modeOptions.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
                updateTrainingModeStatus(this.value);
            }
        });
    });
    
    // Initialize status based on default selection
    const defaultSelected = document.querySelector('.mode-option.selected input[type="radio"]');
    if (defaultSelected) {
        updateTrainingModeStatus(defaultSelected.value);
    }
}

function updateTrainingModeStatus(mode) {
    const statusBadge = document.getElementById('launch-status-badge');
    if (!statusBadge) return;
    
    const modeLabels = {
        'full': '完整蒸馏模式',
        'resume': '断点续训模式',
        'fast': '快速验证模式'
    };
    
    statusBadge.textContent = modeLabels[mode] || '就绪';
    statusBadge.className = 'badge idle';
}

function getCurrentTrainMode() {
    const selected = document.querySelector('input[name="train-mode"]:checked');
    return selected ? selected.value : 'full';
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSliders();
    loadDefaultConfig();
    initFilePicker();
    initTrainingModeSelection();
    managePolling();
});

window.addEventListener('beforeunload', () => {
    if (AppState.logPollInterval) clearInterval(AppState.logPollInterval);
    if (AppState.statusPollInterval) clearInterval(AppState.statusPollInterval);
});
