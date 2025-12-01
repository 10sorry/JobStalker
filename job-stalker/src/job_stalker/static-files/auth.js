/**
 * Terminal Login Interface - Терминал авторизации Telegram
 */

class TelegramAuth {
    constructor() {
        this.authModal = null;
        this.bootComplete = false;
        this.init();
    }

    init() {
        this.checkAuthStatus();
    }

    async checkAuthStatus() {
        try {
            const response = await fetch('/api/auth/status');
            const data = await response.json();

            if (data.status && data.status.authorized) {
                this.onAuthSuccess(data.user);
            } else {
                this.showAuthModal();
            }
        } catch (error) {
            console.error('Auth status check error:', error);
            this.showAuthModal();
        }
    }

    showAuthModal() {
        if (this.authModal) return;

        this.authModal = document.createElement('div');
        this.authModal.id = 'auth-modal';
        this.authModal.className = 'auth-modal';
        this.authModal.innerHTML = `
            <div class="auth-modal-content">
                <div class="terminal-frame">
                    <!-- Terminal Header -->
                    <div class="terminal-header">
                        <div class="terminal-title">
                            TELEGRAM AUTH v1.0
                        </div>
                        <div class="terminal-status">
                            <div class="status-indicator active" id="status-system"></div>
                            <div class="status-indicator" id="status-network"></div>
                            <div class="status-indicator" id="status-auth"></div>
                        </div>
                    </div>

                    <!-- Terminal Main -->
                    <div class="terminal-main">
                        <!-- Left Panel -->
                        <div class="side-panel" id="left-panel">
                            <div class="flask">
                                <div class="flask-container">
                                    <div class="flask-liquid"></div>
                                </div>
                            </div>
                            <div class="chip">
                                <div class="chip-body">
                                    <div class="chip-dot"></div>
                                </div>
                            </div>
                            <div class="led-indicator">
                                <div class="led-dot"></div>
                            </div>
                        </div>

                        <!-- Center Screen -->
                        <div class="center-screen">
                            <!-- Boot Sequence -->
                            <div class="boot-sequence" id="boot-sequence">
                                <div class="boot-line">System Init<span class="boot-ok">[OK]</span></div>
                                <div class="boot-line">Loading Modules<span class="boot-ok">[OK]</span></div>
                                <div class="boot-line">Network Check<span class="boot-ok">[OK]</span></div>
                                <div class="boot-line">Telegram API<span class="boot-ok">[OK]</span></div>
                                <div class="boot-line">Auth System<span class="boot-ok">[OK]</span></div>
                                <div class="boot-ready">&gt; READY_</div>
                            </div>

                            <!-- Auth Screen (скрыт пока загрузка) -->
                            <div class="auth-screen hidden" id="auth-screen">
                                <!-- Phone Form -->
                                <div class="phone-form" id="phone-form">
                                    <div class="screen-header">PHONE AUTH</div>
                                    <div class="screen-subtitle">Enter phone number</div>

                                    <div class="input-display">
                                        <input
                                            type="tel"
                                            id="phone-input"
                                            class="auth-input"
                                            placeholder="+7 XXX XXX XX XX"
                                            autocomplete="off"
                                        >
                                    </div>

                                    <div class="button-panel">
                                        <button class="terminal-btn" id="send-code-btn">
                                            SEND CODE
                                        </button>
                                    </div>

                                    <div class="helper-text">Press ENTER to submit</div>
                                </div>

                                <!-- Code Form (скрыт) -->
                                <div class="code-form hidden" id="code-form">
                                    <div class="screen-header">VERIFICATION</div>
                                    <div class="screen-subtitle">Code sent to Telegram</div>

                                    <div class="input-display">
                                        <input
                                            type="text"
                                            id="code-input"
                                            class="auth-input"
                                            placeholder="12345"
                                            maxlength="5"
                                            autocomplete="off"
                                        >
                                    </div>

                                    <div class="button-panel">
                                        <button class="terminal-btn" id="submit-code-btn">
                                            VERIFY
                                        </button>
                                    </div>

                                    <div class="helper-text">Press ENTER to submit</div>
                                </div>

                                <!-- Password Form (скрыт) -->
                                <div class="password-form hidden" id="password-form">
                                    <div class="screen-header">2FA REQUIRED</div>
                                    <div class="screen-subtitle">Enter password</div>

                                    <div class="input-display">
                                        <input
                                            type="password"
                                            id="password-input"
                                            class="auth-input"
                                            placeholder="Password"
                                            autocomplete="off"
                                        >
                                    </div>

                                    <div class="button-panel">
                                        <button class="terminal-btn" id="submit-password-btn">
                                            UNLOCK
                                        </button>
                                    </div>

                                    <div class="helper-text">Press ENTER to submit</div>
                                </div>

                                <!-- Status -->
                                <div class="auth-status">
                                    <div id="auth-status-text"></div>
                                </div>

                                <!-- Error -->
                                <div class="auth-error hidden" id="auth-error"></div>

                                <!-- Ticker -->
                                <div class="ticker-container">
                                    <div class="ticker-text" id="ticker-text">
                                        SYSTEM READY :: AUTH REQUIRED :: ENTER CREDENTIALS TO CONTINUE :: TELEGRAM API v2.0 :: SECURE CONNECTION ESTABLISHED
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Right Panel -->
                        <div class="side-panel" id="right-panel">
                            <div class="chip">
                                <div class="chip-body">
                                    <div class="chip-dot"></div>
                                </div>
                            </div>
                            <div class="flask">
                                <div class="flask-container">
                                    <div class="flask-liquid"></div>
                                </div>
                            </div>
                            <div class="led-indicator">
                                <div class="led-dot"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(this.authModal);

        // Запускаем boot sequence
        setTimeout(() => {
            this.completeBoot();
        }, 1800);

        this.attachEventListeners();
    }

    completeBoot() {
        const bootSeq = document.getElementById('boot-sequence');
        const authScreen = document.getElementById('auth-screen');

        if (bootSeq && authScreen) {
            bootSeq.style.display = 'none';
            authScreen.classList.remove('hidden');
            this.bootComplete = true;

            // Фокус на первое поле
            document.getElementById('phone-input')?.focus();

            this.setStatusIndicator('network', 'active');
        }
    }


    setStatusIndicator(id, state) {
        const indicator = document.getElementById(`status-${id}`);
        if (indicator) {
            indicator.className = `status-indicator ${state}`;
        }
    }

    updateTicker(text) {
        const ticker = document.getElementById('ticker-text');
        if (ticker) {
            ticker.textContent = text;
        }
    }

    attachEventListeners() {
        // Phone auth
        const sendCodeBtn = document.getElementById('send-code-btn');
        const phoneInput = document.getElementById('phone-input');

        sendCodeBtn?.addEventListener('click', () => this.sendPhoneCode());
        phoneInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendPhoneCode();
        });

        // Code submit
        const submitCodeBtn = document.getElementById('submit-code-btn');
        const codeInput = document.getElementById('code-input');

        submitCodeBtn?.addEventListener('click', () => this.submitCode());
        codeInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.submitCode();
        });

        // Password submit
        const submitPasswordBtn = document.getElementById('submit-password-btn');
        const passwordInput = document.getElementById('password-input');

        submitPasswordBtn?.addEventListener('click', () => this.submitPassword());
        passwordInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.submitPassword();
        });
    }

    async sendPhoneCode() {
        const phone = document.getElementById('phone-input').value.trim();

        if (!phone) {
            this.showError('Phone number required');
            return;
        }

        this.updateStatus('Sending code...');
        this.updateTicker('SENDING AUTH CODE :: PLEASE WAIT');
        this.setStatusIndicator('network', 'waiting');

        try {
            const response = await fetch('/api/auth/phone', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ phone })
            });

            const data = await response.json();

            if (data.success) {
                document.getElementById('phone-form').classList.add('hidden');
                document.getElementById('code-form').classList.remove('hidden');
                this.updateStatus('Code sent to Telegram');
                this.updateTicker('CODE SENT :: CHECK YOUR TELEGRAM APP :: ENTER CODE');
                document.getElementById('code-input')?.focus();
                this.setStatusIndicator('network', 'active');
            } else {
                this.showError(data.error || 'Failed to send code');
                this.setStatusIndicator('network', 'error');
            }
        } catch (error) {
            this.showError('Connection error: ' + error.message);
            this.setStatusIndicator('network', 'error');
        }
    }

    async submitCode() {
        const code = document.getElementById('code-input').value.trim();

        if (!code) {
            this.showError('Code required');
            return;
        }

        this.updateStatus('Verifying code...');
        this.updateTicker('VERIFYING CODE :: PLEASE WAIT');
        this.setStatusIndicator('auth', 'waiting');

        try {
            const response = await fetch('/api/auth/code', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code })
            });

            const data = await response.json();

            if (data.success) {
                this.onAuthSuccess(data.user);
            } else if (data.requires_password) {
                document.getElementById('code-form').classList.add('hidden');
                document.getElementById('password-form').classList.remove('hidden');
                this.updateStatus('2FA password required');
                this.updateTicker('2FA REQUIRED :: ENTER PASSWORD');
                document.getElementById('password-input')?.focus();
            } else {
                this.showError(data.error || 'Invalid code');
                this.setStatusIndicator('auth', 'error');
            }
        } catch (error) {
            this.showError('Connection error: ' + error.message);
            this.setStatusIndicator('auth', 'error');
        }
    }

    async submitPassword() {
        const password = document.getElementById('password-input').value;

        if (!password) {
            this.showError('Password required');
            return;
        }

        this.updateStatus('Verifying password...');
        this.updateTicker('VERIFYING 2FA PASSWORD :: PLEASE WAIT');

        try {
            const response = await fetch('/api/auth/password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ password })
            });

            const data = await response.json();

            if (data.success) {
                this.onAuthSuccess(data.user);
            } else {
                this.showError(data.error || 'Invalid password');
                this.setStatusIndicator('auth', 'error');
            }
        } catch (error) {
            this.showError('Connection error: ' + error.message);
            this.setStatusIndicator('auth', 'error');
        }
    }

    updateStatus(message) {
        const statusEl = document.getElementById('auth-status-text');
        if (statusEl) {
            statusEl.textContent = message;
        }
        document.getElementById('auth-error')?.classList.add('hidden');
    }

    showError(message) {
        const errorEl = document.getElementById('auth-error');
        if (errorEl) {
            errorEl.textContent = message;
            errorEl.classList.remove('hidden');
        }
        this.updateStatus('');
    }

    onAuthSuccess(user) {
        this.updateStatus('✓ AUTH SUCCESS');
        this.updateTicker('AUTHENTICATION SUCCESSFUL :: ACCESS GRANTED :: WELCOME');
        this.setStatusIndicator('auth', 'active');

        setTimeout(() => {
            if (this.authModal) {
                this.authModal.remove();
                this.authModal = null;
            }
            this.displayUserInfo(user);
        }, 1200);
    }

    displayUserInfo(user) {
        console.log('Authorized as:', user);
    }

    // Обработка событий WebSocket
    handleAuthEvent(data) {
        if (data.type === 'auth_status') {
            if (data.status === 'success') {
                this.onAuthSuccess(data.data.user);
            } else if (data.status === 'error') {
                this.showError(data.data.error);
            } else {
                this.updateStatus(this.getStatusMessage(data.status));
            }
        }
    }

    getStatusMessage(status) {
        const messages = {
            'sending_code': 'Sending code...',
            'waiting_code': 'Waiting for code...',
            'verifying_code': 'Verifying code...',
            'waiting_password': 'Password required',
            'verifying_password': 'Verifying password...',
            'success': '✓ Success!'
        };
        return messages[status] || status;
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    window.telegramAuth = new TelegramAuth();
});
