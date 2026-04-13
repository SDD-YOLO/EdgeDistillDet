/* ============================================================
   EdgeDistillDet Web UI - Theme Manager (Light/Dark)
   ============================================================ */

const ThemeManager = {
    currentTheme: 'light',
    STORAGE_KEY: 'edgedistilldet-theme',
    prefersDark: window.matchMedia('(prefers-color-scheme: dark)'),
    
    init() {
        // Load saved preference or use system preference
        const saved = localStorage.getItem(this.STORAGE_KEY);
        if (saved) {
            this.currentTheme = saved;
        } else if (this.prefersDark.matches) {
            this.currentTheme = 'dark';
        }
        
        this.apply(this.currentTheme);
        
        // Listen for system theme changes
        this.prefersDark.addEventListener('change', (e) => {
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                this.apply(e.matches ? 'dark' : 'light');
            }
        });
    },
    
    apply(theme) {
        this.currentTheme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(this.STORAGE_KEY, theme);
        
        // Update all chart themes if charts exist
        if (typeof updateChartsTheme === 'function') {
            updateChartsTheme(theme === 'dark');
        }
        
        // Update log container colors
        const logContainers = document.querySelectorAll('.log-container, .agent-output');
        logContainers.forEach(el => {
            el.style.background = getComputedStyle(document.documentElement).getPropertyValue('--log-bg').trim();
        });
        
        console.log(`[Theme] Applied: ${theme.toUpperCase()}`);
    },
    
    toggle() {
        const next = this.currentTheme === 'light' ? 'dark' : 'light';
        this.apply(next);

        return next;
    },

    getCurrent() { return this.currentTheme; },
    
    isDark() { return this.currentTheme === 'dark'; }
};

// Global toggle function used by the button
function toggleTheme() {
    ThemeManager.toggle();
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
});
