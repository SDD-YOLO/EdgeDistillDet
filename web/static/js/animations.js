/* ============================================================
   EdgeDistillDet Web UI - 动画效果系统
   ============================================================ */

const Animations = (() => {
    // 检测用户是否偏好减少动画
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const EASE_OUT = 'cubic-bezier(0.16, 1, 0.3, 1)';
    const STAGGER_DELAY = 80;

    // ==================== 工具函数 ====================
    function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
    function lerp(a, b, t) { return a + (b - a) * t; }

    function isDarkMode() {
        return document.documentElement.getAttribute('data-theme') === 'dark';
    }

    function getCSSVar(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    // ==================== 1. 页面加载动画 (Stagger Fade-In + Slide-Up) ====================
    const PageLoadAnimator = {
        async init() {
            if (prefersReducedMotion) {
                this._showImmediate();
                return;
            }

            const selectors = [
                '.md-app-bar',
                '.tab-nav',
                '.train-launcher .launch-info',
                '.train-launcher .launch-modes',
                '.train-launcher .launch-actions',
                '.config-card',
                '.log-card',
                '.quick-stats .stat-card',
                '.metrics-overview .overview-card',
                '.chart-card'
            ];

            let delayIndex = 0;
            for (const selector of selectors) {
                const els = document.querySelectorAll(selector);
                els.forEach((el, i) => {
                    el.style.opacity = '0';
                    el.style.transform = 'translateY(24px)';
                    el.style.willChange = 'opacity, transform';
                    el.style.transition = 'none';
                    el.dataset.staggerIdx = delayIndex + i;
                    el.classList.add('anim-stagger-target');
                });
                delayIndex += els.length;
            }

            // 触发重排后启动动画
            requestAnimationFrame(() => {
                document.body.offsetHeight; // force reflow
                this._playStagger();
            });
        },

        _showImmediate() {
            document.querySelectorAll('.anim-stagger-target').forEach(el => {
                el.style.opacity = '';
                el.style.transform = '';
                el.style.willChange = '';
            });
        },

        _playStagger() {
            const targets = [...document.querySelectorAll('.anim-stagger-target')]
                .sort((a, b) => (a.dataset.staggerIdx || 0) - (b.dataset.staggerIdx || 0));

            targets.forEach((el, idx) => {
                setTimeout(() => {
                    el.style.transition = `opacity 500ms ${EASE_OUT}, transform 500ms ${EASE_OUT}`;
                    el.style.opacity = '1';
                    el.style.transform = 'translateY(0)';
                    setTimeout(() => { el.style.willChange = ''; }, 520);
                }, idx * STAGGER_DELAY);
            });
        }
    };

    // ==================== 2. 训练按钮状态动画 ====================
    const TrainingButtonAnim = {
        states: ['idle', 'loading', 'success', 'error'],
        currentState: 'idle',

        setState(state) {
            if (!this.states.includes(state)) return;
            this.currentState = state;
            const btn = document.getElementById('btn-start-training');
            if (!btn) return;

            btn.classList.remove('btn-state-idle', 'btn-state-loading', 'btn-state-success', 'btn-state-error');
            btn.classList.add(`btn-state-${state}`);

            switch (state) {
                case 'loading':
                    this._enterLoading(btn);
                    break;
                case 'success':
                    this._enterSuccess(btn);
                    break;
                case 'error':
                    this._enterError(btn);
                    break;
                default:
                    this._resetToIdle(btn);
            }
        },

        _enterLoading(btn) {
            btn.disabled = true;
            btn.innerHTML = `
                <span class="btn-spinner"></span>
                <span>训练中...</span>
            `;
            btn.style.pointerEvents = 'none';
        },

        _enterSuccess(btn) {
            btn.disabled = false;
            btn.style.pointerEvents = '';
            btn.innerHTML = `
                <span class="material-icons btn-check-icon">check_circle</span>
                <span>训练完成</span>
            `;
            btn.classList.add('btn-success-flash');
            // 触发 reflow 以重新播放动画
            void btn.offsetWidth;
            btn.classList.remove('btn-success-flash');
            void btn.offsetWidth;
            btn.classList.add('btn-success-flash');

            // 3秒后恢复 idle
            setTimeout(() => this.setState('idle'), 3500);
        },

        _enterError(btn) {
            btn.disabled = false;
            btn.style.pointerEvents = '';
            btn.innerHTML = `
                <span class="material-icons" style="font-size:18px">error</span>
                <span>开始训练</span>
            `;
            btn.classList.add('btn-error-shake');
            setTimeout(() => btn.classList.remove('btn-error-shake'), 600);
        },

        _resetToIdle(btn) {
            btn.disabled = false;
            btn.style.pointerEvents = '';
            btn.innerHTML = `
                <span class="material-icons">play_arrow</span>
                <span>开始训练</span>
            `;
        }
    };

    // ==================== 3. Loss曲线实时平滑动画增强 ====================
    const ChartAnimator = {
        init() {
            // 增强 Chart.js 的默认动画配置
            if (typeof Chart !== 'undefined') {
                Chart.defaults.animation.duration = 300;
                Chart.defaults.animation.easing = 'easeOutCubic';
            }
        },

        // 平滑推送新数据点到图表
        pushChartData(chart, datasetIndex, newValue, maxPoints = 100) {
            if (!chart) return;

            const ds = chart.data.datasets[datasetIndex];
            if (!ds) return;

            ds.data.push(newValue);

            // 自动生成 label（epoch 编号）
            const nextEpoch = chart.data.labels.length + 1;
            chart.data.labels.push(nextEpoch.toString());

            // 限制数据点数量，实现滑动窗口
            while (ds.data.length > maxPoints) {
                ds.data.shift();
                chart.data.labels.shift();
            }

            chart.update({
                mode: 'active',
                duration: 300,
                easing: 'easeOutCubic'
            });
        },

        // 批量更新图表数据（带渐变填充）
        updateWithGradient(chart, datasetsData, labels) {
            if (!chart) return;
            chart.data.labels = labels;
            chart.data.datasets.forEach((ds, i) => {
                if (datasetsData[i] !== undefined) {
                    ds.data = datasetsData[i];
                }
            });

            // 确保填充效果启用
            chart.data.datasets.forEach(ds => {
                ds.fill = true;
            });

            chart.update({
                duration: 300,
                easing: 'easeOutCubic'
            });
        }
    };

    // ==================== 4. 知识蒸馏流动画 (Canvas粒子系统) ====================
    const DistillFlowCanvas = {
        canvas: null,
        ctx: null,
        particles: [],
        animId: null,
        running: false,
        paused: false,
        kdLoss: 3.0,
        alpha: 0.5,
        scale: 1.0,
        mouseX: -1,
        mouseY: -1,
        isHovering: false,

        init(containerId = 'distill-flow-canvas') {
            const container = document.getElementById(containerId);
            if (!container) return;

            this.canvas = document.createElement('canvas');
            this.canvas.id = 'distill-canvas';
            this.canvas.width = container.offsetWidth || 600;
            this.canvas.height = Math.min(200, window.innerHeight * 0.25);
            this.canvas.style.cursor = 'pointer';
            container.appendChild(this.canvas);
            this.ctx = this.canvas.getContext('2d');

            // ResizeObserver 响应容器变化
            this._resizeObserver = new ResizeObserver(entries => {
                for (const entry of entries) {
                    this.canvas.width = entry.contentRect.width;
                    this.canvas.height = Math.min(200, window.innerHeight * 0.25);
                }
                if (!this.running && !this.paused) {
                    this._createParticles();
                    if (!prefersReducedMotion) this._drawStatic();
                }
            });
            this._resizeObserver.observe(container);

            // ---- 交互事件绑定 ----
            // 点击暂停/继续
            this.canvas.addEventListener('click', () => this.togglePause());

            // 悬停显示信息
            this.canvas.addEventListener('mouseenter', () => { this.isHovering = true; });
            this.canvas.addEventListener('mouseleave', () => {
                this.isHovering = false;
                this.mouseX = -1;
                this.mouseY = -1;
            });
            this.canvas.addEventListener('mousemove', (e) => {
                const rect = this.canvas.getBoundingClientRect();
                this.mouseX = e.clientX - rect.left;
                this.mouseY = e.clientY - rect.top;
            });

            // 滚轮缩放
            this.canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                this.scale = Math.max(0.4, Math.min(3, this.scale + delta));
            }, { passive: false });

            this._createParticles();
            if (!prefersReducedMotion) {
                this.running = true;
                this._animate();
            } else {
                this._drawStatic();
            }
        },

        setKDLoss(loss) {
            this.kdLoss = Math.max(0.1, Math.min(10, loss));
        },

        setAlpha(a) {
            this.alpha = Math.max(0, Math.min(1, a));
        },

        start() {
            if (this.running || this.paused) return;
            this.running = true;
            if (!prefersReducedMotion) this._animate();
        },

        stop() {
            this.running = false;
            this.paused = false;
            if (this.animId) {
                cancelAnimationFrame(this.animId);
                this.animId = null;
            }
        },

        togglePause() {
            if (!this.running && !this.paused) return; // 未启动时不可暂停
            this.paused = !this.paused;
            if (this.paused) {
                this.running = false;
                if (this.animId) {
                    cancelAnimationFrame(this.animId);
                    this.animId = null;
                }
                // 绘制暂停状态
                this._drawPausedFrame();
            } else {
                this.running = true;
                this._animate();
            }
        },

        destroy() {
            this.stop();
            if (this._resizeObserver) this._resizeObserver.disconnect();
            if (this.canvas && this.canvas.parentNode) {
                this.canvas.parentNode.removeChild(this.canvas);
            }
        },

        _createParticles() {
            this.particles = [];
            const count = prefersReducedMotion ? 20 : 60;
            const w = this.canvas.width;
            const h = this.canvas.height;
            const teacherX = w * 0.18;
            const studentX = w * 0.82;
            const cy = h / 2;

            for (let i = 0; i < count; i++) {
                const progress = Math.random();
                this.particles.push({
                    x: lerp(teacherX + 30, studentX - 30, progress),
                    y: cy + (Math.random() - 0.5) * h * 0.6,
                    vx: 0.8 + Math.random() * 1.5,
                    vy: (Math.random() - 0.5) * 0.8,
                    size: 2 + Math.random() * 3,
                    progress,
                    opacity: 0.4 + Math.random() * 0.6,
                    phase: Math.random() * Math.PI * 2
                });
            }
        },

        _getParticleColor(particle) {
            const lossNorm = Math.min(1, this.kdLoss / 6);
            const hue = lerp(180, 15, lossNorm);
            const sat = 70 + this.alpha * 30;
            const light = 50 + particle.opacity * 20;
            return `hsla(${hue.toFixed(0)}, ${sat.toFixed(0)}%, ${light.toFixed(0)}%, ${(particle.opacity * 0.85).toFixed(2)})`;
        },

        _drawModelBox(label, x, y, w, h, color, iconChar) {
            const ctx = this.ctx;
            ctx.save();

            ctx.shadowColor = `rgba(0,0,0,0.12)`;
            ctx.shadowBlur = 12;
            ctx.shadowOffsetY = 3;

            const r = 10;
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();

            const grad = ctx.createLinearGradient(x, y, x, y + h);
            grad.addColorStop(0, color);
            grad.addColorStop(1, this._darkenColor(color, 0.15));
            ctx.fillStyle = grad;
            ctx.fill();

            ctx.restore();

            ctx.fillStyle = '#fff';
            ctx.font = "bold 13px 'Plus Jakarta Sans', sans-serif";
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(iconChar, x + w / 2, y + h * 0.35);
            ctx.font = "11px 'Noto Sans SC', sans-serif";
            ctx.fillText(label, x + w / 2, y + h * 0.7);

            // 悬停高亮检测（教师/学生模型盒子）
            if (this.isHovering && this.mouseX >= 0) {
                if (this.mouseX >= x && this.mouseX <= x + w && this.mouseY >= y && this.mouseY <= y + h) {
                    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            }
        },

        _darkenColor(hex, amount) {
            if (hex.startsWith('#')) {
                let r = parseInt(hex.slice(1, 3), 16);
                let g = parseInt(hex.slice(3, 5), 16);
                let b = parseInt(hex.slice(5, 7), 16);
                r = Math.max(0, Math.floor(r * (1 - amount)));
                g = Math.max(0, Math.floor(g * (1 - amount)));
                b = Math.max(0, Math.floor(b * (1 - amount)));
                return `rgb(${r},${g},${b})`;
            }
            return hex;
        },

        _drawTooltip(ctx, w, h) {
            if (!this.isHovering || this.mouseX < 0) return;
            const mx = this.mouseX, my = this.mouseY;

            const lines = [
                `KD Loss: ${this.kdLoss.toFixed(4)}`,
                `Alpha: ${this.alpha.toFixed(4)}`,
                `缩放: ${(this.scale * 100).toFixed(0)}%`,
                `${this.paused ? '已暂停 (点击继续)' : '运行中 (点击暂停)'}`
            ];

            const padding = 8;
            const lineHeight = 16;
            const tw = 140;
            const th = padding * 2 + lines.length * lineHeight;

            let tx = mx + 14, ty = my - th / 2 - 8;
            if (tx + tw > w) tx = mx - tw - 14;
            if (ty < 4) ty = 4;
            if (ty + th > h - 4) ty = h - th - 4;

            // Tooltip 背景
            ctx.save();
            ctx.fillStyle = isDarkMode() ? 'rgba(28,28,32,0.95)' : 'rgba(255,255,255,0.96)';
            ctx.shadowColor = 'rgba(0,0,0,0.18)';
            ctx.shadowBlur = 12;
            ctx.beginPath();
            ctx.roundRect(tx, ty, tw, th, 8);
            ctx.fill();
            ctx.restore();

            // 文字
            ctx.font = "12px 'Plus Jakarta Sans', monospace";
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            lines.forEach((line, i) => {
                ctx.fillStyle = i === lines.length - 1
                    ? (isDarkMode() ? '#7dd5d4' : '#006a6b')
                    : (isDarkMode() ? '#e0e3e2' : '#1a1c1c');
                ctx.fillText(line, tx + padding, ty + padding + i * lineHeight);
            });
        },

        _drawPauseOverlay(ctx, w, h) {
            ctx.save();
            ctx.globalAlpha = 0.35;
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, w, h);
            ctx.restore();

            // 暂停图标 (双竖线)
            const cx = w / 2, cy = h / 2;
            const barW = 6, barH = 24;
            ctx.fillStyle = isDarkMode() ? '#e0e3e2' : '#1a1c1c';
            ctx.beginPath();
            ctx.roundRect(cx - barW - 4, cy - barH / 2, barW, barH, 3);
            ctx.fill();
            ctx.beginPath();
            ctx.roundRect(cx + 4, cy - barH / 2, barW, barH, 3);
            ctx.fill();

            ctx.fillStyle = isDarkMode() ? 'rgba(224,227,226,0.55)' : 'rgba(26,28,28,0.45)';
            ctx.font = "12px 'Noto Sans SC', sans-serif";
            ctx.textAlign = 'center';
            ctx.fillText('点击继续动画', cx, cy + barH / 2 + 20);
        },

        _animate() {
            if (!this.running || this.paused) return;

            const ctx = this.ctx;
            const w = this.canvas.width;
            const h = this.canvas.height;
            const teacherX = w * 0.18;
            const studentX = w * 0.82;
            const boxW = 56;
            const boxH = 44;
            const cy = h / 2;
            const time = performance.now() * 0.001;

            ctx.clearRect(0, 0, w, h);

            // 应用缩放变换
            if (Math.abs(this.scale - 1) > 0.01) {
                ctx.save();
                ctx.translate(w / 2, h / 2);
                ctx.scale(this.scale, this.scale);
                ctx.translate(-w / 2, -h / 2);
            }

            // 背景流线
            ctx.save();
            ctx.globalAlpha = 0.06;
            for (let i = 0; i < 3; i++) {
                const yOffset = (i - 1) * h * 0.25;
                ctx.beginPath();
                ctx.moveTo(teacherX + boxW + 10, cy + yOffset);
                ctx.bezierCurveTo(
                    w * 0.35, cy + yOffset + Math.sin(time + i) * 15,
                    w * 0.65, cy + yOffset - Math.sin(time * 0.7 + i) * 15,
                    studentX - 10, cy + yOffset
                );
                ctx.strokeStyle = getCSSVar('--md-primary') || '#006a6b';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
            ctx.restore();

            // 绘制模型盒子
            const teacherColor = '#6750A4';
            const studentColor = '#006a6b';
            this._drawModelBox('教师模型', teacherX - boxW / 2, cy - boxH / 2, boxW, boxH, teacherColor, 'T');
            this._drawModelBox('学生模型', studentX - boxW / 2, cy - boxH / 2, boxW, boxH, studentColor, 'S');

            // 更新并绘制粒子
            this.particles.forEach(p => {
                p.progress += p.vx * 0.003 * (0.8 + this.alpha);

                if (p.progress > 1) {
                    p.progress = 0;
                    p.x = teacherX + boxW / 2 + 10;
                    p.y = cy + (Math.random() - 0.5) * h * 0.6;
                    p.vy = (Math.random() - 0.5) * 0.8;
                    p.phase = Math.random() * Math.PI * 2;
                }

                const t = p.progress;
                const startX = teacherX + boxW / 2 + 10;
                const endX = studentX - boxW / 2 - 10;
                const cp1x = w * 0.35, cp1y = cy + Math.sin(p.phase + time) * h * 0.18;
                const cp2x = w * 0.65, cp2y = cy - Math.sin(p.phase * 0.7 + time * 0.8) * h * 0.18;

                const mt = 1 - t;
                p.x = mt * mt * mt * startX + 3 * mt * mt * t * cp1x + 3 * mt * t * t * cp2x + t * t * t * endX;
                p.y = mt * mt * mt * cy + 3 * mt * mt * t * cp1y + 3 * mt * t * t * cp2y + t * t * t * cy;
                p.y += Math.sin(time * 2 + p.phase) * 3;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = this._getParticleColor(p);
                ctx.fill();

                // 发光效果
                ctx.save();
                ctx.globalAlpha = p.opacity * 0.3;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size * 2.5, 0, Math.PI * 2);
                ctx.fillStyle = this._getParticleColor(p);
                ctx.filter = 'blur(4px)';
                ctx.fill();
                ctx.restore();
                ctx.filter = 'none';
            });

            // 中间标签
            ctx.fillStyle = isDarkMode() ? 'rgba(224,227,226,0.55)' : 'rgba(26,28,28,0.45)';
            ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
            ctx.textAlign = 'center';
            ctx.fillText(`KD Loss: ${this.kdLoss.toFixed(3)}  |  Alpha: ${this.alpha.toFixed(2)} | 点击暂停`, w / 2, h - 12);

            // 恢复缩放变换
            if (Math.abs(this.scale - 1) > 0.01) {
                ctx.restore();
            }

            // Tooltip 在缩放外绘制（不受缩放影响坐标）
            this._drawTooltip(ctx, w, h);

            this.animId = requestAnimationFrame(() => this._animate());
        },

        _drawPausedFrame() {
            if (!this.ctx) return;
            const ctx = this.ctx;
            const w = this.canvas.width;
            const h = this.canvas.height;

            ctx.clearRect(0, 0, w, h);

            // 绘制当前帧的静态内容
            if (Math.abs(this.scale - 1) > 0.01) {
                ctx.save();
                ctx.translate(w / 2, h / 2);
                ctx.scale(this.scale, this.scale);
                ctx.translate(-w / 2, -h / 2);
            }

            const teacherX = w * 0.18, studentX = w * 0.82;
            const bw = 56, bh = 44, cy = h / 2;
            this._drawModelBox('教师模型', teacherX - bw / 2, cy - bh / 2, bw, bh, '#6750A4', 'T');
            this._drawModelBox('学生模型', studentX - bw / 2, cy - bh / 2, bw, bh, '#006a6b', 'S');

            // 绘制静态粒子快照
            this.particles.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = this._getParticleColor(p);
                ctx.globalAlpha = p.opacity * 0.7;
                ctx.fill();
            });
            ctx.globalAlpha = 1;

            if (Math.abs(this.scale - 1) > 0.01) ctx.restore();

            this._drawPauseOverlay(ctx, w, h);
            this._drawTooltip(ctx, w, h);
        },

        _drawStatic() {
            const ctx = this.ctx;
            const w = this.canvas.width;
            const h = this.canvas.height;
            ctx.clearRect(0, 0, w, h);

            const teacherX = w * 0.18, studentX = w * 0.82;
            const bw = 56, bh = 44, cy = h / 2;
            this._drawModelBox('教师模型', teacherX - bw / 2, cy - bh / 2, bw, bh, '#6750A4', 'T');
            this._drawModelBox('学生模型', studentX - bw / 2, cy - bh / 2, bw, bh, '#006a6b', 'S');

            ctx.strokeStyle = isDarkMode() ? 'rgba(125,213,212,0.5)' : 'rgba(0,106,107,0.4)';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.beginPath();
            ctx.moveTo(teacherX + bw / 2 + 15, cy);
            ctx.lineTo(studentX - bw / 2 - 15, cy);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    };

    // ==================== 5. 数字滚动动画 (CountUp) ====================
    const CountUpAnim = {
        activeAnimations: new Map(),

        animate(elementId, targetValue, options = {}) {
            const el = document.getElementById(elementId);
            if (!el) return;

            const duration = options.duration || 800;
            const decimals = options.decimals ?? 2;
            const prefix = options.prefix || '';
            const suffix = options.suffix || '';

            // 取消同一元素上正在运行的动画
            if (this.activeAnimations.has(elementId)) {
                cancelAnimationFrame(this.activeAnimations.get(elementId).rafId);
            }

            const startValue = parseFloat(el.dataset.currentValue) || 0;
            const startTime = performance.now();

            const tick = (now) => {
                const elapsed = now - startTime;
                const progress = Math.min(elapsed / duration, 1);
                // easeOutExpo 缓动函数
                const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
                const current = lerp(startValue, targetValue, eased);

                el.textContent = `${prefix}${current.toFixed(decimals)}${suffix}`;
                el.dataset.currentValue = current;

                if (progress < 1) {
                    this.activeAnimations.get(elementId).rafId = requestAnimationFrame(tick);
                } else {
                    this.activeAnimations.delete(elementId);
                }
            };

            this.activeAnimations.set(elementId, { rafId: requestAnimationFrame(tick) });
        },

        animateStat(statKey, value) {
            const idMap = {
                'stat-loss': { decimals: 4 },
                'stat-map50': { decimals: 2 },
                'stat-map95': { decimals: 2 },
                'stat-lr': { decimals: 5, suffix: '' },
                'ov-map50': { suffix: '%', decimals: 2 },
                'ov-fps': { suffix: ' fps', decimals: 0 },
                'ov-params': { suffix: 'M', decimals: 1 }
            };
            const opts = idMap[statKey] || { decimals: 2 };
            this.animate(statKey, parseFloat(value), opts);
        },

        cancelAll() {
            this.activeAnimations.forEach(v => cancelAnimationFrame(v.rafId));
            this.activeAnimations.clear();
        }
    };

    // ==================== 6. 训练完成庆祝动画 (Confetti) ====================
    const Celebration = {
        particles: [],
        canvas: null,
        ctx: null,
        animId: null,
        badgeEl: null,

        trigger(options = {}) {
            if (prefersReducedMotion) {
                this._showBadgeOnly(options.message || '训练完成！');
                return;
            }

            this._createOverlay();
            this._fireConfetti();
            this._showBadge(options.message || '训练完成！');

            // 自动清理
            setTimeout(() => this._cleanup(), 4500);
        },

        _createOverlay() {
            // 创建全屏 canvas 用于 confetti
            this.canvas = document.createElement('canvas');
            this.canvas.id = 'celebration-canvas';
            this.canvas.style.cssText =
                'position:fixed;top:0;left:0;width:100vw;height:100vh;' +
                'pointer-events:none;z-index:9999;';
            document.body.appendChild(this.canvas);
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
            this.ctx = this.canvas.getContext('2d');

            // 背景闪烁
            this._flashBackground();
        },

        _flashBackground() {
            const flash = document.createElement('div');
            flash.id = 'celebration-flash';
            flash.style.cssText =
                'position:fixed;top:0;left:0;width:100%;height:100%;' +
                'pointer-events:none;z-index:9998;background:' +
                'radial-gradient(circle at 50% 40%,' +
                'rgba(0,212,170,0.15),transparent 60%);' +
                'animation:celebrationFlash 1.5s ease-out forwards;';
            document.body.appendChild(flash);
        },

        _fireConfetti() {
            const colors = ['#00D4AA', '#FFD700', '#FF6B6B', '#6C63FF', '#4ECDC4', '#FF9F43', '#A78BFA'];
            const w = this.canvas.width;
            const h = this.canvas.height;
            const cx = w / 2;
            const cy = h * 0.35;

            this.particles = [];
            for (let i = 0; i < prefersReducedMotion ? 30 : 140; i++) {
                const angle = (Math.PI * 2 * (i / 140)) + (Math.random() - 0.5) * 0.5;
                const speed = 4 + Math.random() * 10;
                this.particles.push({
                    x: cx + (Math.random() - 0.5) * 40,
                    y: cy + (Math.random() - 0.5) * 20,
                    vx: Math.cos(angle) * speed,
                    vy: Math.sin(angle) * speed - Math.random() * 6,
                    size: 4 + Math.random() * 7,
                    rotation: Math.random() * 360,
                    rotSpeed: (Math.random() - 0.5) * 15,
                    color: colors[i % colors.length],
                    shape: Math.random() > 0.5 ? 'rect' : 'circle',
                    opacity: 1,
                    gravity: 0.15 + Math.random() * 0.08,
                    friction: 0.99
                });
            }

            this._animateConfetti();
        },

        _animateConfetti() {
            if (!this.ctx) return;
            const ctx = this.ctx;
            const w = this.canvas.width;
            const h = this.canvas.height;

            ctx.clearRect(0, 0, w, h);

            let alive = false;
            this.particles.forEach(p => {
                if (p.opacity <= 0) return;
                alive = true;

                p.vy += p.gravity;
                p.vx *= p.friction;
                p.x += p.vx;
                p.y += p.vy;
                p.rotation += p.rotSpeed;

                // 边界检测
                if (p.y > h + 20) {
                    p.opacity -= 0.05;
                }
                if (p.x < -20 || p.x > w + 20) {
                    p.opacity -= 0.05;
                }

                ctx.save();
                ctx.translate(p.x, p.y);
                ctx.rotate(p.rotation * Math.PI / 180);
                ctx.globalAlpha = Math.max(0, p.opacity);
                ctx.fillStyle = p.color;

                if (p.shape === 'rect') {
                    ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
                } else {
                    ctx.beginPath();
                    ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
                    ctx.fill();
                }
                ctx.restore();
            });

            if (alive) {
                this.animId = requestAnimationFrame(() => this._animateConfetti());
            } else {
                this._cleanup();
            }
        },

        _showBadge(message) {
            const badge = document.createElement('div');
            badge.id = 'celebration-badge';
            badge.innerHTML = `
                <div class="celebration-badge-inner">
                    <span class="material-icons badge-icon">emoji_events</span>
                    <span class="badge-text">${message}</span>
                </div>
            `;
            document.body.appendChild(badge);
            this.badgeEl = badge;

            // 徽章缩放旋转动画由 CSS 处理
            requestAnimationFrame(() => {
                badge.classList.add('show');
            });
        },

        _showBadgeOnly(message) {
            this._showBadge(message);
            setTimeout(() => this._cleanup(), 2500);
        },

        _cleanup() {
            if (this.animId) cancelAnimationFrame(this.animId);
            if (this.canvas && this.canvas.parentNode) this.canvas.parentNode.removeChild(this.canvas);
            const flash = document.getElementById('celebration-flash');
            if (flash) flash.parentNode.removeChild(flash);
            const badge = document.getElementById('celebration-badge');
            if (badge) {
                badge.classList.add('hiding');
                setTimeout(() => { if (badge.parentNode) badge.parentNode.removeChild(badge); }, 400);
            }
            this.particles = [];
            this.canvas = null;
            this.ctx = null;
            this.badgeEl = null;
        }
    };

    // ==================== 7. 数据增强预览动画 ====================
    const AugmentationPreview = {
        currentTransforms: [],
        intervalId: null,
        isPlaying: false,

        init(containerId = 'aug-preview-container') {
            const container = document.getElementById(containerId);
            if (!container) return;

            this.container = container;
            this.imageEl = container.querySelector('.aug-preview-image') || this._createPreviewImage(container);
            this.labelEl = container.querySelector('.aug-label');
            this.transforms = [
                { name: '原图', css: '' },
                { name: '水平翻转', css: 'scaleX(-1)' },
                { name: '垂直翻转', css: 'scaleY(-1)' },
                { name: '旋转 90°', css: 'rotate(90deg)' },
                { name: '旋转 180°', css: 'rotate(180deg)' },
                { name: '缩放 1.3x', css: 'scale(1.3)' },
                { name: '缩放 0.7x', css: 'scale(0.7)' },
                { name: '透视倾斜', css: 'perspective(300px) rotateX(10deg) rotateY(-5deg)' },
                { name: '亮度增强', css: 'brightness(1.3)' },
                { name: '对比度增强', css: 'contrast(1.4)' },
                { name: '饱和度增强', css: 'saturate(1.6)' },
                { name: '模糊模拟', css: 'blur(1.5px)' },
                { name: '灰度化', css: 'grayscale(1)' },
                { name: '色相旋转', css: 'hue-rotate(90deg)' },
                { name: '反相', css: 'invert(1)' },
            ];
            this.currentIndex = 0;

            // 悬停时自动播放变换序列
            container.addEventListener('mouseenter', () => this.startPlayback());
            container.addEventListener('mouseleave', () => this.stopPlayback());

            // 点击切换下一个变换
            container.addEventListener('click', () => {
                this.nextTransform();
                this.stopPlayback(); // 手动点击时停止自动播放
            });
        },

        _createPreviewImage(container) {
            const img = document.createElement('div');
            img.className = 'aug-preview-image';
            img.innerHTML = `<svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg" width="120" height="80">
                <defs>
                    <linearGradient id="augGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#006a6b;stop-opacity:0.7"/>
                        <stop offset="100%" style="stop-color:#2a6fb5;stop-opacity:0.7"/>
                    </linearGradient>
                </defs>
                <rect width="120" height="80" rx="6" fill="url(#augGrad)"/>
                <text x="60" y="36" text-anchor="middle" fill="#fff" font-size="11" font-family="sans-serif">Sample</text>
                <text x="60" y="52" text-anchor="middle" fill="rgba(255,255,255,0.7)" font-size="9" font-family="sans-serif">Image</text>
                <!-- 模拟目标框 -->
                <rect x="35" y="18" width="22" height="16" rx="2" fill="none" stroke="#FFD700" stroke-width="1.5"/>
                <rect x="68" y="32" width="14" height="24" rx="2" fill="none" stroke="#FF6B6B" stroke-width="1.5"/>
            </svg>`;
            container.appendChild(img);

            const label = document.createElement('div');
            label.className = 'aug-label';
            label.textContent = '原图';
            container.appendChild(label);

            return img;
        },

        applyTransform(index) {
            if (!this.imageEl || !this.transforms[index]) return;
            this.currentIndex = index;
            const tf = this.transforms[index];

            // 使用 CSS transition 实现平滑过渡
            if (!prefersReducedMotion) {
                this.imageEl.style.transition = 'transform 0.35s ease, filter 0.35s ease';
            }
            this.imageEl.style.transform = tf.css || 'none';
            this.imageEl.style.filter = tf.css.includes('brightness') || tf.css.includes('contrast')
                || tf.css.includes('saturate') || tf.css.includes('blur')
                || tf.css.includes('grayscale') || tf.css.includes('hue-rotate')
                || tf.css.includes('invert') ? tf.css : '';

            if (tf.css.includes('perspective')) {
                // perspective 变换需要特殊处理
                this.imageEl.style.transform = tf.css;
            } else if (tf.css.includes('rotate') || tf.css.includes('scale')) {
                this.imageEl.style.transform = tf.css;
                this.imageEl.style.filter = '';
            }

            if (this.labelEl) this.labelEl.textContent = tf.name;
        },

        nextTransform() {
            this.currentIndex = (this.currentIndex + 1) % this.transforms.length;
            this.applyTransform(this.currentIndex);
        },

        prevTransform() {
            this.currentIndex = (this.currentIndex - 1 + this.transforms.length) % this.transforms.length;
            this.applyTransform(this.currentIndex);
        },

        startPlayback() {
            if (this.isPlaying || prefersReducedMotion) return;
            this.isPlaying = true;
            this.intervalId = setInterval(() => this.nextTransform(), 900);
        },

        stopPlayback() {
            this.isPlaying = false;
            if (this.intervalId) {
                clearInterval(this.intervalId);
                this.intervalId = null;
            }
        },

        setImage(src) {
            if (this.imageEl) {
                if (src.startsWith('data:') || src.startsWith('http')) {
                    this.imageEl.innerHTML = `<img src="${src}" alt="预览" style="width:100%;height:auto;display:block;border-radius:6px;" />`;
                }
            }
        }
    };

    // ==================== 公共 API ====================
    return {
        PageLoadAnimator,
        TrainingButtonAnim,
        ChartAnimator,
        DistillFlowCanvas,
        CountUpAnim,
        Celebration,
        AugmentationPreview,

        // ==================== Prompt 4: 增强组件 ====================

        /* ---- 增强进度条 (条纹动画 + 完成闪烁) ---- */
        ProgressBar: {
            update(elementId, current, total, options = {}) {
                const bar = document.getElementById(elementId);
                if (!bar) return;

                const percent = total > 0 ? Math.min(100, (current / total) * 100) : 0;
                bar.style.width = `${percent.toFixed(1)}%`;

                const wrapper = bar.closest('.progress-bar-wrapper');
                if (wrapper && !prefersReducedMotion) {
                    if (percent >= 99.5 && percent < 100) {
                        wrapper.classList.add('progress-near-complete');
                        wrapper.classList.remove('progress-complete');
                    } else if (percent >= 100) {
                        wrapper.classList.remove('progress-near-complete');
                        wrapper.classList.add('progress-complete');
                        setTimeout(() => wrapper.classList.remove('progress-complete'), 2000);
                    } else {
                        wrapper.classList.remove('progress-near-complete', 'progress-complete');
                    }
                }
            },

            setStripesActive(elementId, active) {
                const wrapper = document.getElementById(elementId)?.closest('.progress-bar-wrapper') ||
                                 document.getElementById(`${elementId}-wrapper`);
                if (wrapper) {
                    wrapper.classList.toggle('stripes-active', active);
                }
            }
        },

        /* ---- 训练日志搜索/过滤增强 ---- */
        LogEnhancer: {
            init(containerId = 'log-container') {
                this.container = document.getElementById(containerId);
                if (!this.container) return;

                // 创建工具栏
                const toolbar = document.createElement('div');
                toolbar.className = 'log-toolbar';
                toolbar.innerHTML = `
                    <input type="text" class="log-search-input" placeholder="搜索日志..." />
                    <div class="log-filter-chips">
                        <button class="chip log-filter-btn active" data-level="all">全部</button>
                        <button class="chip log-filter-btn" data-level="info">信息</button>
                        <button class="chip log-filter-btn" data-level="epoch">Epoch</button>
                        <button class="chip log-filter-btn" data-level="warning">警告</button>
                        <button class="chip log-filter-btn" data-level="error">错误</button>
                        <button class="chip log-filter-btn" data-level="success">成功</button>
                        <button class="btn-icon-sm log-copy-btn" title="复制所有日志">
                            <span class="material-icons" style="font-size:16px">content_copy</span>
                        </button>
                    </div>
                `;
                this.container.parentNode.insertBefore(toolbar, this.container);

                // 搜索功能
                const searchInput = toolbar.querySelector('.log-search-input');
                searchInput.addEventListener('input', (e) => this.filter(e.target.value));

                // 过滤按钮
                toolbar.querySelectorAll('.log-filter-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        toolbar.querySelectorAll('.log-filter-btn').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                        this.setFilter(btn.dataset.level);
                    });
                });

                // 复制按钮
                toolbar.querySelector('.log-copy-btn').addEventListener('click', () => this.copyLogs());
            },

            filter(query) {
                query = query.toLowerCase().trim();
                if (!query || !this.container) return;
                const lines = this.container.querySelectorAll('.log-line');
                let visibleCount = 0;
                lines.forEach(line => {
                    const text = line.textContent.toLowerCase();
                    const match = text.includes(query);
                    line.style.display = match ? '' : 'none';
                    if (match) visibleCount++;
                });
                this._updateCount(visibleCount, lines.length);
            },

            setFilter(level) {
                if (!this.container) return;
                const lines = this.container.querySelectorAll('.log-line');
                let visibleCount = 0;
                lines.forEach(line => {
                    if (level === 'all' || line.classList.contains(level)) {
                        line.style.display = '';
                        visibleCount++;
                    } else {
                        line.style.display = 'none';
                    }
                });
                this._updateCount(visibleCount, lines.length);
            },

            copyLogs() {
                if (!this.container) return;
                const text = [...this.container.querySelectorAll('.log-line')]
                    .map(el => el.textContent).join('\n');
                navigator.clipboard.writeText(text).then(() => {
                    showToast('日志已复制到剪贴板', 'success');
                }).catch(() => {
                    // fallback for older browsers
                    const ta = document.createElement('textarea');
                    ta.value = text; ta.style.position = 'fixed'; ta.left = '-9999px';
                    document.body.appendChild(ta); ta.select();
                    document.execCommand('copy'); document.body.removeChild(ta);
                    showToast('日志已复制到剪贴板', 'success');
                });
            },

            _updateCount(visible, total) {
                let counter = this.container.parentNode.querySelector('.log-counter');
                if (!counter && visible !== total) {
                    counter = document.createElement('span');
                    counter.className = 'log-counter';
                    counter.style.cssText =
                        'font-size:10px;color:var(--md-on-surface-variant);padding:2px 6px;';
                    this.container.appendChild(counter);
                }
                if (counter) {
                    counter.textContent = `显示 ${visible}/${total} 条`;
                }
            }
        },

        // ==================== Prompt 3: WebSocket 实时通信层 ====================
        RealTimeSocket: {
            ws: null,
            reconnectAttempts: 0,
            maxReconnects: 10,
            handlers: {},
            connectionState: 'disconnected',

            connect(url = null) {
                const wsUrl = url || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/training`;

                try {
                    this.ws = new WebSocket(wsUrl);
                    this.connectionState = 'connecting';
                    this._onStateChange?.({ state: 'connecting' });

                    this.ws.onopen = () => {
                        console.log('[WS] Connected to training stream');
                        this.reconnectAttempts = 0;
                        this.connectionState = 'connected';
                        this._onStateChange?.({ state: 'connected' });
                        this._send({ type: 'ping' });
                    };

                    this.ws.onmessage = (event) => {
                        try {
                            const msg = JSON.parse(event.data);
                            this._dispatch(msg);
                        } catch (e) {
                            // Raw text message
                            this._dispatch({ type: 'raw', data: event.data });
                        }
                    };

                    this.ws.onclose = (e) => {
                        console.log(`[WS] Connection closed (code: ${e.code})`);
                        this.connectionState = 'disconnected';
                        this._onStateChange?.({ state: 'disconnected', code: e.code });
                        this._scheduleReconnect();
                    };

                    this.ws.onerror = (err) => {
                        console.error('[WS] Error:', err);
                        this.connectionState = 'error';
                        this._onStateChange?.({ state: 'error' });
                    };
                } catch (e) {
                    console.error('[WS] Failed to connect:', e);
                    // Fallback: use SSE polling mode
                    this._fallbackToSSE();
                }
            },

            disconnect() {
                this.maxReconnects = 0; // prevent reconnect
                if (this.ws) { this.ws.close(); this.ws = null; }
            },

            send(data) { this._send(data); },
            _send(data) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify(data));
                }
            },

            on(type, handler) {
                if (!this.handlers[type]) this.handlers[type] = [];
                this.handlers[type].push(handler);
            },

            off(type, handler) {
                if (this.handlers[type]) {
                    this.handlers[type] = this.handlers[type].filter(h => h !== handler);
                }
            },

            getState() { return this.connectionState; },
            onStateChange(cb) { this._onStateChange = cb; },

            _dispatch(msg) {
                // 全局消息处理器
                (this.handlers['*'] || []).forEach(h => h(msg));
                // 类型特定处理器
                const typeHandlers = this.handlers[msg.type] || [];
                typeHandlers.forEach(h => h(msg.data || msg));
            },

            _scheduleReconnect() {
                if (this.reconnectAttempts >= this.maxReconnects) {
                    console.warn('[WS] Max reconnect attempts reached.');
                    return;
                }
                const delay = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 30000);
                this.reconnectAttempts++;
                console.log(`[WS] Reconnecting in ${Math.round(delay / 1000)}s... (attempt ${this.reconnectAttempts})`);
                setTimeout(() => this.connect(), delay);
            },

            _fallbackToSSE() {
                console.log('[WS] Falling back to SSE polling mode');
                this.connectionState = 'sse';
                this._onStateChange?.({ state: 'sse' });
            }

        },

        prefersReducedMotion,

        // 初始化所有动画
        initAll(options = {}) {
            PageLoadAnimator.init();
            ChartAnimator.init();

            // 可选初始化
            if (options.enableLogSearch !== false) {
                // 延迟初始化，确保 DOM 已就绪
                requestAnimationFrame(() => LogEnhancer.init());
            }

            if (options.enableWebSocket !== false) {
                // 自动连接 WebSocket（如果后端支持）
                setTimeout(() => {
                    RealTimeSocket.connect();
                }, 1500);
            }
        }
    };
})();
