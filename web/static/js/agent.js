/* ============================================================
   EdgeDistillDet Web UI - AI Agent Interface
   ============================================================ */

// Agent State
const AgentState = {
    conversationHistory: [],
    availableActions: [],
    isProcessing: false,
    apiUrl: '',
    apiKey: '',
    // ==================== Extension Registry ====================
    // 预留的扩展注册接口，后续可动态添加新功能
    registeredExtensions: {
        // 示例: 
        // 'custom_action': {
        //     name: '自定义动作',
        //     description: '描述',
        //     handler: async (params) => { ... },
        //     enabled: true
        // }
    }
};

// ==================== Initialize Agent ====================
async function initAgent() {
    loadAgentApiConfig();
    AgentState.availableActions = [
        'auto_tune',
        'analyze_result',
        'suggest_config',
        'compare_models'
    ];
    if (AgentState.apiUrl) {
        showToast(`Agent API 已配置: ${AgentState.apiUrl}`, 'success');
    } else {
        showToast('请先配置外部 Agent API 地址，然后才能调用 Agent 功能。', 'info');
    }
}

// ==================== Action Dispatcher ====================
/**
 * 调用 Agent 动作
 * @param {string} action - 动作名称
 */
async function callAgentAction(action) {
    if (AgentState.isProcessing) {
        showToast('Agent 正在处理中，请稍候...', 'warning');
        return;
    }

    syncAgentApiConfig();
    if (!AgentState.apiUrl) {
        showToast('请先在 Agent API 面板中填写外部 API 地址', 'warning');
        return;
    }

    // 检查是否是预留扩展槽位
    if (action.startsWith('slot_') || action.includes('extension')) {
        showToast('该功能即将推出，敬请期待!', 'info');
        return;
    }

    const actionNames = {
        'auto_tune': '自动超参数调优',
        'analyze_result': '训练结果分析',
        'suggest_config': '智能配置推荐',
        'compare_models': '模型对比分析'
    };

    addAgentMessage('user', `启动 ${actionNames[action] || action}...`);
    
    try {
        AgentState.isProcessing = true;
        showAgentThinking();

        const headers = getAgentApiRequestHeaders();
        const response = await fetch(AgentState.apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...headers
            },
            body: JSON.stringify({ action: action, params: {} })
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || response.statusText || 'Agent API 返回错误');
        }

        let responseText = '';
        if (typeof result === 'string') {
            responseText = result;
        } else if (result && (result.reply || result.message)) {
            responseText = result.reply || result.message;
        }

        removeAgentThinking();
        if (responseText) {
            addAgentMessage('agent', responseText);
        }
        updateAgentOutput(formatOutputForPanel(result, action));
        
    } catch (error) {
        removeAgentThinking();
        addAgentMessage('agent', `抱歉，处理请求时出错: ${error.message}`);
        updateAgentOutput(`Error: ${error.message}`);
    } finally {
        AgentState.isProcessing = false;
    }
}

// ==================== Chat Interface ====================
const AgentChat = {
    input: null,
    sendButton: null,
    messageContainer: null,
    isReady: false,
};

function initAgentChat() {
    AgentChat.input = document.getElementById('agent-input');
    AgentChat.sendButton = document.getElementById('agent-send-btn');
    AgentChat.messageContainer = document.getElementById('agent-chat-messages');

    if (!AgentChat.input || !AgentChat.sendButton || !AgentChat.messageContainer) {
        console.error('[Agent] 聊天组件未正确初始化');
        return;
    }

    AgentChat.sendButton.addEventListener('click', sendAgentMessage);
    AgentChat.input.addEventListener('keydown', handleChatKeydown);
    AgentChat.input.addEventListener('input', () => autoResizeTextarea(AgentChat.input));
    AgentChat.input.addEventListener('compositionend', () => autoResizeTextarea(AgentChat.input));

    autoResizeTextarea(AgentChat.input);
    AgentChat.isReady = true;
}

function sendAgentMessage() {
    if (!AgentChat.isReady) return;
    if (AgentState.isProcessing) {
        showToast('Agent 正在处理中，请稍候...', 'warning');
        return;
    }

    const rawText = AgentChat.input.value.replace(/\u00A0/g, ' ');
    const text = rawText.trim();
    if (!text) {
        AgentChat.input.focus();
        return;
    }

    addAgentMessage('user', text);
    AgentChat.input.value = '';
    autoResizeTextarea(AgentChat.input);
    AgentChat.input.focus();

    syncAgentApiConfig();
    if (AgentState.apiUrl) {
        forwardUserMessageToApi(text).catch(error => {
            addAgentMessage('agent', `抱歉，处理请求时出错: ${error.message}`);
            updateAgentOutput(`Error: ${error.message}`);
        });
    } else {
        showToast('请先配置外部 Agent API 地址，然后再发送消息。', 'warning');
    }
}

function handleChatKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendAgentMessage();
        return;
    }

    if (event.key === 'Escape') {
        AgentChat.input.blur();
        return;
    }
}

function autoResizeTextarea(textarea) {
    if (!textarea) return;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, 160);
    textarea.style.height = `${Math.max(newHeight, 48)}px`;
}

/**
 * 处理自然语言查询（智能路由）
 */
async function processNaturalQuery(query) {
    AgentState.isProcessing = true;
    showAgentThinking();

    try {
        // 简单的关键词匹配路由（后续可替换为 LLM）
        const lowerQuery = query.toLowerCase();
        let response = '';
        let actionType = null;

        if (lowerQuery.includes('调优') || lowerQuery.includes('tun') || lowerQuery.includes('优化')) {
            actionType = 'auto_tune';
            response = `收到您的调优请求。我将基于当前配置进行超参数搜索。

**建议方案：**
1. **学习率**: 当前 lr0=0.01，建议尝试 [0.005, 0.02] 范围
2. **蒸馏权重**: alpha 范围建议 [0.3, 0.7]，配合温度退火
3. **数据增强**: 当前 mosaic=0.8，对于小目标检测建议提高到 1.0

> 💡 提示：点击左侧"自动超参数调优"按钮可启动完整调优流程`;
        }
        else if (lowerQuery.includes('分析') || lowerQuery.includes('analy') || lowerQuery.includes('结果')) {
            actionType = 'analyze_result';
            response = `正在分析训练结果...

**当前模型表现摘要：**
- 📊 mAP@50: 92.34% (最佳)
- 📊 mAP@50-95: 68.45%
- ⚡ 推理速度: ~85 FPS (RTX 3090)
- 🎯 小目标召回率: 78.2%

**改进建议：**
1. 蒸馏损失在 epoch 80 后趋于平稳，可适当增加 focal_gamma
2. 温度退火曲线正常，T_min 可进一步降低到 1.0 以增强后期锐化

详细分析请查看指标监控 Tab 中的图表。`;
        }
        else if (lowerQuery.includes('推荐') || lowerQuery.includes('suggest') || lowerQuery.includes('建议') || lowerQuery.includes('config')) {
            actionType = 'suggest_config';
            response = `根据您的数据集特征，我推荐以下配置：

\`\`\`yaml
distillation:
  alpha_init: 0.55      # 中等初始权重
  T_max: 7.0             # 较高起始温度
  T_min: 1.0             # 较低终止温度以增强区分度
  w_kd: 0.55             # 略增KD权重
  scale_boost: 2.5       # 增强小目标关注
  
training:
  epochs: 200           # 增加轮数以充分收敛
  imgsz: 640            # 保持标准尺寸
  mosaic: 1.0           # 全强度马赛克增强
  close_mosaic: 15      # 较早关闭以稳定后期训练
\`\`\`

> 点击"应用此配置"可将上述设置填入表单`;
        }
        else if (lowerQuery.includes('对比') || lowerQuery.includes('compar') || lowerQuery.includes('model')) {
            actionType = 'compare_models';
            response = `📊 **模型性能对比报告**

| 模型 | mAP@50 | Params | FPS(GPU) | FPS(CPU) |
|------|--------|--------|----------|----------|
| YOLOv5n (原生) | 72.3% | 2.5M | 142 | 28 |
| YOLO26n (蒸馏前) | 81.5% | 3.2M | 128 | 24 |
| **YOLO26n (蒸馏后)** | **92.3%** | **3.2M** | **125** | **23** |

✅ **关键发现：**
- 蒸馏后精度提升 +13.2%，参数量不变
- GPU推理速度仅下降 2.3%（几乎无影响）
- 边缘设备适配性良好

💡 建议部署蒸馏后的模型到边缘端`;
        }
        else if (lowerQuery.includes('你好') || lowerQuery.includes('hello') || lowerQuery.includes('hi')) {
            response = `你好！我是 EdgeDistillDet AI Agent 🤖

我可以帮助你完成以下任务：
- 🔧 **自动调优**: 搜索最优超参数组合
- 📈 **结果分析**: 深入解读训练指标
- 💡 **配置推荐**: 智能生成训练配置  
- ⚖️ **模型对比**: 多维度性能评估

请选择左侧的功能卡片，或直接告诉我你需要什么帮助!`;
        }
        else if (lowerQuery.includes('帮助') || lowerQuery.includes('help') || lowerQuery.includes('?')) {
            response = `**EdgeDistillDet Web UI 使用指南**

**🔧 训练配置 Tab:**
- 配置学生/教师模型路径和所有蒸馏参数
- 调整滑块实时预览参数值
- 支持加载/保存/重置配置文件

**📊 指标监控 Tab:**
- 实时查看 Loss、mAP、学习率等曲线
- 导出图表为 PNG 格式
- 查看训练结果摘要表格

**🤖 AI Agent Tab:**
- 使用自然语言交互
- 一键调用高级功能
- 扩展接口支持自定义插件`;
        }
        else {
            response =
                `我理解你的问题是："${query}"。\n\n` +
                `当前 Agent 支持以下能力：\n` +
                `- 超参数自动调优\n` +
                `- 训练结果深度分析\n` +
                `- 智能配置推荐\n` +
                `- 多模型性能对比\n\n` +
                `你可以输入“帮我调优参数”或“分析训练结果”来快速触发。`;
        }

        await simulateDelay(800); // Natural typing delay
        removeAgentThinking();
        addAgentMessage('agent', response);
        
        // Update output panel
        updateAgentOutput(JSON.stringify({ action: actionType, query, timestamp: new Date().toISOString() }, null, 2));

    } catch (error) {
        removeAgentThinking();
        addAgentMessage('agent', `处理时出现错误: ${error.message}`);
    } finally {
        AgentState.isProcessing = false;
    }
}

function syncAgentApiConfig() {
    const urlInput = document.getElementById('agent-api-url');
    const keyInput = document.getElementById('agent-api-key');
    if (urlInput) {
        AgentState.apiUrl = urlInput.value.trim();
    }
    if (keyInput) {
        AgentState.apiKey = keyInput.value.trim();
    }
}

function getAgentApiRequestHeaders() {
    const headers = {};
    if (AgentState.apiKey) {
        headers['Authorization'] = AgentState.apiKey;
    }
    return headers;
}

function loadAgentApiConfig() {
    try {
        const saved = window.localStorage.getItem('edge_distill_agent_api');
        if (saved) {
            const config = JSON.parse(saved);
            AgentState.apiUrl = config.apiUrl || '';
            AgentState.apiKey = config.apiKey || '';
        }
    } catch (error) {
        console.warn('加载 Agent API 配置失败', error);
    }

    const urlInput = document.getElementById('agent-api-url');
    const keyInput = document.getElementById('agent-api-key');
    if (urlInput) urlInput.value = AgentState.apiUrl;
    if (keyInput) keyInput.value = AgentState.apiKey;
}

function saveAgentApiConfig() {
    const urlInput = document.getElementById('agent-api-url');
    const keyInput = document.getElementById('agent-api-key');
    if (!urlInput) return;

    AgentState.apiUrl = urlInput.value.trim();
    AgentState.apiKey = keyInput ? keyInput.value.trim() : '';
    window.localStorage.setItem('edge_distill_agent_api', JSON.stringify({ apiUrl: AgentState.apiUrl, apiKey: AgentState.apiKey }));
    showToast('Agent API 配置已保存', 'success');
}

async function testAgentApi() {
    syncAgentApiConfig();
    if (!AgentState.apiUrl) {
        showToast('请先填写 Agent API 地址', 'warning');
        return;
    }

    try {
        const response = await fetch(AgentState.apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAgentApiRequestHeaders()
            },
            body: JSON.stringify({ action: 'ping', params: {} })
        });
        if (!response.ok) {
            const result = await response.text();
            throw new Error(`HTTP ${response.status} ${response.statusText}: ${result}`);
        }
        showToast('Agent API 连接成功', 'success');
    } catch (error) {
        showToast(`Agent API 连接失败: ${error.message}`, 'error');
        console.error(error);
    }
}

async function forwardUserMessageToApi(query) {
    const headers = getAgentApiRequestHeaders();
    const response = await fetch(AgentState.apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...headers
        },
        body: JSON.stringify({ query: query })
    });

    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || response.statusText || 'Agent API 返回错误');
    }

    const reply = typeof result === 'string' ? result : (result.reply || result.message || JSON.stringify(result, null, 2));
    removeAgentThinking();
    addAgentMessage('agent', reply);
    updateAgentOutput(formatOutputForPanel(result, 'user_query'));
}

// ==================== Message Management ====================
function addAgentMessage(role, content) {
    const container = document.getElementById('agent-chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${role}`;
    
    const avatarClass = role === 'agent' ? 'agent-avatar' : 'user-avatar';
    const icon = role === 'agent' ? 'smart_toy' : 'person';
    
    msgDiv.innerHTML = `
        <div class="message-avatar ${avatarClass}">
            <span class="material-icons">${icon}</span>
        </div>
        <div class="message-content">
            <div>${formatMarkdown(content)}</div>
        </div>
    `;
    
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;

    AgentState.conversationHistory.push({ role, content, timestamp: Date.now() });
}

function clearAgentChat() {
    const container = document.getElementById('agent-chat-messages');
    container.innerHTML = '';
    AgentState.conversationHistory = [];
}

function showAgentThinking() {
    const container = document.getElementById('agent-chat-messages');
    const thinkingDiv = document.createElement('div');
    thinkingDiv.id = 'thinking-indicator';
    thinkingDiv.className = 'chat-message agent';
    thinkingDiv.innerHTML = `
        <div class="message-avatar agent-avatar">
            <span class="material-icons">smart_toy</span>
        </div>
        <div class="message-content">
            <div class="thinking-dots">
                <span></span><span></span><span></span>
                正在思考中...
            </div>
        </div>
    `;
    container.appendChild(thinkingDiv);
    container.scrollTop = container.scrollHeight;
}

function removeAgentThinking() {
    const el = document.getElementById('thinking-indicator');
    if (el) el.remove();
}

// ==================== Output Panel ====================
function updateAgentOutput(content) {
    const panel = document.getElementById('agent-output-content');
    panel.textContent = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
}

// ==================== Extension Registration Interface ====================
/**
 * 注册新的 Agent 功能扩展
 * @param {Object} extension - 扩展定义对象
 * @param {string} extension.id - 唯一标识符
 * @param {string} extension.name - 显示名称
 * @param {string} extension.description - 功能描述
 * @param {Function} handler - 处理函数
 * 
 * 使用示例:
 * registerAgentExtension({
 *   id: 'my_custom_tool',
 *   name: '我的自定义工具',
 *   description: '这是一个自定义功能',
 *   handler: async (params) => { ... }
 * });
 */
function registerAgentExtension(extension) {
    if (!extension.id || !extension.name || typeof extension.handler !== 'function') {
        console.error('Invalid extension format. Required: id, name, handler function');
        return false;
    }

    AgentState.registeredExtensions[extension.id] = {
        name: extension.name,
        description: extension.description || '',
        handler: extension.handler,
        enabled: true,
        registeredAt: new Date().toISOString()
    };

    console.log(`[Agent] Extension registered: ${extension.name}`);
    
    // 可以在这里动态更新UI
    updateExtensionSlots();
    
    return true;
}

/**
 * 注销已注册的扩展
 */
function unregisterAgentExtension(extensionId) {
    if (AgentState.registeredExtensions[extensionId]) {
        delete AgentState.registeredExtensions[extensionId];
        console.log(`[Agent] Extension unregistered: ${extensionId}`);
        updateExtensionSlots();
        return true;
    }
    return false;
}

/**
 * 更新UI中的扩展插槽显示
 */
function updateExtensionSlots() {
    const extensions = Object.values(AgentState.registeredExtensions);
    const slots = ['agent-slot-1', 'agent-slot-2'];
    
    extensions.slice(0, 2).forEach((ext, idx) => {
        const slot = document.getElementById(slots[idx]);
        if (slot) {
            slot.className = 'agent-action-card';
            slot.onclick = () => handleExtensionAction(ext);
            slot.querySelector('.action-name').textContent = ext.name;
            slot.querySelector('.action-desc').textContent = ext.description;
            slot.querySelector('.action-icon').textContent = 'extension'; // or custom icon
            slot.querySelector('.badge').remove(); // Remove coming-soon badge
        }
    });
}

/**
 * 处理来自扩展的动作
 */
async function handleExtensionAction(extension) {
    addAgentMessage('user', `调用扩展功能: ${extension.name}`);
    
    try {
        AgentState.isProcessing = true;
        showAgentThinking();
        
        const result = await extension.handler({});
        
        removeAgentThinking();
        addAgentMessage('agent', result.message || JSON.stringify(result, null, 2));
        updateAgentOutput(JSON.stringify({
            source: extension.id,
            result,
            timestamp: new Date().toISOString()
        }, null, 2));
        
    } catch (error) {
        removeAgentThinking();
        addAgentMessage('agent', `扩展执行错误: ${error.message}`);
    } finally {
        AgentState.isProcessing = false;
    }
}

// ==================== Response Generators ====================
function generateAutoTuneResponse(result) {
    return `## 🔧 自动超参数调优

${result.status === 'pending' ? 
    '> ⚠️ 完整调优引擎开发中，以下是模拟演示...' : ''}

### 调优策略
采用 **贝叶斯优化 (Bayesian Optimization)** 结合 **网格搜索**:

1. **第一阶段**: 粗粒度网格搜索确定大致范围
2. **第二阶段**: 贝叶斯优化精细调优 Top-K 组合  
3. **第三阶段**: 微小扰动验证稳定性

### 建议参数组
| 参数 | 当前值 | 建议值 | 变化 |
|------|--------|--------|------|
| alpha_init | 0.50 | **0.58** | +16% |
| T_max | 6.0 | **7.5** | +25% |
| w_kd | 0.50 | **0.62** | +24% |
| lr0 | 0.01 | **0.007** | -30% |

### 预期收益
- mAP 提升: **+2.1% ~ +4.5%** (置信区间 95%)
- 收敛速度: 快约 15-25 epochs
- 稳定性: 方差降低 ~18%

> ✨ 点击"应用此配置"一键应用最优参数`;
}

function generateAnalyzeResponse(result) {
    return `## 📊 训练结果深度分析

### 整体评价
⭐ **评级: A-** (优秀)

**优势:**
- ✅ 收敛曲线平滑，无明显震荡
- ✅ 蒸馏效果显著，学生模型接近教师水平
- ✅ 小目标检测能力提升明显 (scale_boost 生效)

**待改进:**
- ⚠️ epoch 100+ 后 loss 出现轻微过拟合迹象
- ⚠️ 温度退火后期梯度较小，可考虑降低 T_min

### 详细指标分解

**损失函数:**
- Box Loss 最终: 0.3589 ↓ (较基线 -42%)
- CLS Loss 最终: 0.3012 ↓ (较基线 -38%)
- DFL Loss 最终: 0.9145 ↓ (较基线 -31%)

**检测指标:**
- Precision: 0.8590 (平衡较好)
- Recall: 0.8156 (可尝试提升)
- mAP@50: 0.9187 (优秀)
- mAP@50-95: 0.6789 (有提升空间)

### 下一步建议
1. 尝试将 warm_epochs 从 5 → 8
2. 增加 mixup 到 0.15 以提升泛化
3. 考虑使用 cosine LR 替代线性衰减`;
}

function generateSuggestConfigResponse(result) {
    return `## 💡 智能配置推荐

基于数据集分析和历史经验生成的配置:

\`\`\`yaml
# === 推荐蒸馏配置 ===
distillation:
  student_weight: "path/to/student.pt"
  teacher_weight: "path/to/teacher.pt"
  alpha_init: 0.55          # 稍微提高初始权重
  T_max: 7.0                 # 更高的起始温度
  T_min: 1.0                 # 更低的终止温度(增强区分)
  warm_epochs: 8             # 更长的预热
  w_kd: 0.60                 # 增强知识蒸馏
  w_focal: 0.25              # 略微减少focal权重
  w_feat: 0.10               # 启用特征对齐
  scale_boost: 2.5           # 强化小目标
  focal_gamma: 2.5           # 更强的聚焦

# === 推荐训练配置 ===
training:
  data_yaml: "data.yaml"
  device: 0
  epochs: 200               # 更多轮次
  imgsz: 640
  batch: 16                  # 固定batch大小
  workers: 12
  lr0: 0.008                 # 较低的学习率
  lrf: 0.05                  # 更低的最终lr因子
  warmup_epochs: 5
  mosaic: 1.0                # 全强度
  mixup: 0.15                # 增加mixup
  close_mosaic: 12           # 较早关闭mosaic
  amp: true
\`\`\`

> 此配置针对**小目标密集场景**优化`;
}

function generateCompareModelsResponse(result) {
    return `## ⚖️ 模型对比分析报告

### 对比矩阵

| 维度 | 原生YOLO | 蒸馏前 | **蒸馏后** | 提升 |
|------|----------|--------|------------|------|
| **精度** |
| mAP@50 | 72.3% | 81.5% | **92.3%** | **+27.4%** |
| mAP@50-95 | 48.2% | 56.8% | **68.4%** | **+41.9%** |
| Precision | 79.1% | 83.2% | **86.8%** | +9.7% |
| Recall | 71.5% | 78.9% | **82.3%** | +15.1% |
| **效率** |
| Params | 2.5M | 3.2M | **3.2M** | =0% |
| FLOPs | 6.4G | 8.1G | **8.1G** | =0% |
| FPS (GPU) | 142 | 128 | **125** | -2.3% |
| FPS (CPU) | 28 | 24 | **23** | -4.2% |
| **边缘部署** |
| RK3588 FPS | 18 | 14 | **13.5** | -3.6% |
| 内存占用 | 245MB | 312MB | **312MB** | =0% |

### 关键结论

🏆 **蒸馏在几乎零成本的情况下大幅提升精度！**

1. **精度提升显著**: mAP@50 从 72.3% → 92.3%, 提升超过 **20个百分点**
2. **参数量不增加**: 学生模型结构未变，仅通过知识迁移提升能力
3. **推理速度影响微小**: GPU 仅降速 2.3%, CPU 降速 4.2%
4. **边缘友好**: 在 RK3588 上仍能保持实时帧率

### 建议
✅ **推荐部署蒸馏后模型**用于生产环境`;
}

// ==================== Utilities ====================
function formatMarkdown(text) {
    // Basic markdown to HTML conversion
    return text
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/^### (.+)$/gm, '<h4 style="margin:12px 0 6px;font-size:14px">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 style="margin:14px 0 8px;font-size:15px;color:var(--md-primary)">$1</h3>')
        .replace(/^# (.+)$/gm, '<h2 style="margin:16px 0 10px">$1</h2>')
        .replace(/^\- (.+)$/gm, '<li style="margin-left:16px;margin-bottom:4px">$1</li>')
        .replace(/^\| .+$/gm, match => `<div style="overflow-x:auto">${match}</div>`)
        .replace(/^> (.+)$/gm, '<blockquote style="border-left:3px solid var(--md-primary);padding:6px 14px;background:var(--md-primary-container);border-radius:0 8px 8px 0;margin:8px 0;font-size:12px">$1</blockquote>')
        .replace(/\n/g, '<br>');
}

function formatOutputForPanel(data, action) {
    return `╔══════════════════════════════════════╗
║  Agent Action Report
║  Timestamp: ${new Date().toLocaleString()}
╠══════════════════════════════════════╣
║  Action: ${action}
║  Status: ${data.status}
╚══════════════════════════════════════╝

${JSON.stringify(data, null, 2)}
`;
}

function simulateDelay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initAgent();
    initAgentChat();
});

// Expose registration function globally for external extensions
window.registerAgentExtension = registerAgentExtension;
window.unregisterAgentExtension = unregisterAgentExtension;
