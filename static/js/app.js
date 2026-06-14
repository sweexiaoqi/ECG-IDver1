// Set this to your deployed Render backend URL when deploying the frontend separately to Netlify.
// Example: const BACKEND_URL = "https://ecg-id-backend.onrender.com";
const BACKEND_URL = "";

// State Management
let currentScreen = 'screen-main';
let loginFiles = [];
let registerFiles = [];
let devToken = localStorage.getItem('dev_token') || '';
let dashboardInterval = null;
let performanceChart = null;

// Temporary cache of last uploaded files for auto-registration from denial screen
let lastUploadedFiles = [];
let lastAttemptedUsername = '';

// DOM Elements
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toast-message');
const toastIcon = document.getElementById('toast-icon');

// -------------------------------------------------------------
// NAVIGATION AND SCREEN ROUTING
// -------------------------------------------------------------

function showScreen(screenId) {
    // Hide current screen
    const currentEl = document.getElementById(currentScreen);
    if (currentEl) {
        currentEl.classList.remove('active');
        // Small delay to let fade-out animate before display:none
        setTimeout(() => {
            currentEl.style.display = 'none';
        }, 150);
    }

    // Stop dashboard polling if leaving dev portal
    if (screenId !== 'screen-dev-dashboard' && dashboardInterval) {
        clearInterval(dashboardInterval);
        dashboardInterval = null;
    }

    // Show new screen
    const newEl = document.getElementById(screenId);
    if (newEl) {
        currentScreen = screenId;
        setTimeout(() => {
            newEl.style.display = 'block';
            // Force reflow
            newEl.offsetHeight;
            newEl.classList.add('active');
        }, 150);
    }

    // Initializations based on screen
    if (screenId === 'screen-dev-dashboard') {
        initDashboard();
    } else if (screenId === 'screen-login') {
        clearUploadZone('login');
        loadSamplesList('login');
    } else if (screenId === 'screen-register') {
        clearUploadZone('register');
        loadSamplesList('register');
        if (lastAttemptedUsername) {
            document.getElementById('register-username').value = lastAttemptedUsername;
        }
    }
}

function goToVerifyRegistered() {
    showScreen('screen-login');
    // Pre-select files if we have them from auto-register
    if (lastUploadedFiles.length > 0) {
        loginFiles = [...lastUploadedFiles];
        renderFileList('login');
    }
}

// -------------------------------------------------------------
// TOAST NOTIFICATIONS
// -------------------------------------------------------------

function showToast(message, type = 'info') {
    toastMessage.textContent = message;
    
    // Set icon based on type
    if (type === 'success') {
        toastIcon.textContent = 'check_circle';
        toastIcon.style.color = '#00C48C';
    } else if (type === 'error') {
        toastIcon.textContent = 'error';
        toastIcon.style.color = '#FF4D4D';
    } else {
        toastIcon.textContent = 'info';
        toastIcon.style.color = '#6C47FF';
    }
    
    toast.classList.add('active');
    setTimeout(() => {
        toast.classList.remove('active');
    }, 4000);
}

// -------------------------------------------------------------
// DRAG & DROP & FILE UPLOADS
// -------------------------------------------------------------

function setupDragAndDrop(type) {
    const zone = document.getElementById(`${type}-upload-zone`);
    const input = document.getElementById(`${type}-file-input`);

    if (!zone || !input) return;

    // Drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        zone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            zone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    zone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = Array.from(dt.files);
        handleFilesSelection(type, files);
    }, false);

    // Handle file dialog selection
    input.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleFilesSelection(type, files);
    });
}

function handleFilesSelection(type, files) {
    if (type === 'login') {
        loginFiles = [...loginFiles, ...files];
        // Remove duplicates
        loginFiles = loginFiles.filter((file, index, self) =>
            index === self.findIndex((t) => t.name === file.name)
        );
        renderFileList('login');
    } else {
        registerFiles = [...registerFiles, ...files];
        registerFiles = registerFiles.filter((file, index, self) =>
            index === self.findIndex((t) => t.name === file.name)
        );
        renderFileList('register');
    }
}

function removeFile(type, index) {
    if (type === 'login') {
        loginFiles.splice(index, 1);
        renderFileList('login');
    } else {
        registerFiles.splice(index, 1);
        renderFileList('register');
    }
}

function renderFileList(type) {
    const listContainer = document.getElementById(`${type}-files-list`);
    const files = type === 'login' ? loginFiles : registerFiles;
    
    listContainer.innerHTML = '';
    
    files.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'file-item';
        
        let icon = 'description';
        if (file.name.endsWith('.hea')) icon = 'settings_applications';
        if (file.name.endsWith('.dat')) icon = 'binary_data';
        
        item.innerHTML = `
            <div class="file-info">
                <span class="material-symbols-rounded">${icon}</span>
                <span>${file.name}</span>
            </div>
            <span class="material-symbols-rounded file-remove" onclick="removeFile('${type}', ${index})">close</span>
        `;
        listContainer.appendChild(item);
    });
}

function clearUploadZone(type) {
    if (type === 'login') {
        loginFiles = [];
        renderFileList('login');
    } else {
        registerFiles = [];
        renderFileList('register');
    }
}

// -------------------------------------------------------------
// DATASET SAMPLE DOWNLOAD UTILITIES
// -------------------------------------------------------------

async function loadSamplesList(type) {
    const container = document.getElementById(`${type}-samples-list`);
    if (!container) return;
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/samples/list`);
        const files = await response.json();
        
        if (files.length === 0) {
            container.innerHTML = '<span class="loading-label">No test records available yet. Wait for auto-generation.</span>';
            return;
        }
        
        container.innerHTML = '';
        
        // Group files by subject for cleaner rendering
        // E.g., Person_01_rec_1.hea and Person_01_rec_1.dat
        const baseNames = new Set();
        files.forEach(f => {
            const base = f.substring(0, f.lastIndexOf('.'));
            if (base) baseNames.add(base);
        });
        
        baseNames.forEach(base => {
            const button = document.createElement('button');
            button.className = 'sample-dl-btn';
            button.innerHTML = `
                <span>${base} (WFDB pair)</span>
                <span class="material-symbols-rounded">download</span>
            `;
            button.onclick = () => downloadSamplePair(base);
            container.appendChild(button);
        });
        
    } catch (error) {
        console.error('Error fetching samples:', error);
        container.innerHTML = '<span class="loading-label">Failed to load test samples.</span>';
    }
}

async function downloadSamplePair(baseName) {
    showToast(`Downloading files for ${baseName}...`);
    try {
        // WFDB requires both .hea and .dat files
        const filesToDownload = [`${baseName}.hea`, `${baseName}.dat`];
        
        for (const filename of filesToDownload) {
            const link = document.createElement('a');
            link.href = `${BACKEND_URL}/api/samples/download/${filename}`;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            // Small pause between downloads
            await new Promise(resolve => setTimeout(resolve, 300));
        }
        showToast(`Downloaded .hea and .dat pair for ${baseName}. Upload both together to authenticate!`, 'success');
    } catch (e) {
        showToast(`Failed to download sample: ${e}`, 'error');
    }
}

// Initialize drag and drop events on load
setupDragAndDrop('login');
setupDragAndDrop('register');

// -------------------------------------------------------------
// API CALLS: AUTHENTICATION AND REGISTRATION
// -------------------------------------------------------------

// Verify User
document.getElementById('btn-authenticate').addEventListener('click', async () => {
    if (loginFiles.length === 0) {
        showToast('Please upload an ECG recording file (.csv, .txt, or a .hea/.dat pair) first.', 'error');
        return;
    }
    
    const btn = document.getElementById('btn-authenticate');
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-rounded logo-icon">autorenew</span> AUTHENTICATING...`;
    
    const formData = new FormData();
    loginFiles.forEach(file => {
        formData.append('files', file);
    });
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/auth/verify`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Authentication service failed.');
        }

        // Cache files and result details in case of denial and auto-registration
        lastUploadedFiles = [...loginFiles];
        lastAttemptedUsername = result.username !== 'Unregistered' && result.username !== 'Unknown' ? result.username : '';

        if (result.verified) {
            // Approved
            document.getElementById('approved-accuracy').textContent = `${(result.accuracy * 100).toFixed(2)}%`;
            document.getElementById('approved-username').textContent = result.username;
            document.getElementById('approved-desc').textContent = result.description;
            showScreen('screen-auth-approved');
        } else {
            // Denied
            document.getElementById('denied-accuracy').textContent = `${(result.accuracy * 100).toFixed(2)}%`;
            document.getElementById('denied-desc').textContent = result.description;
            
            // Setup auto-register container
            const autoRegContainer = document.getElementById('denied-autoregister-container');
            const suggestedUserEl = document.getElementById('denied-suggested-username');
            const btnLabelEl = document.getElementById('btn-register-username-label');
            
            // If the failure was 'User not found' or we can suggest registration
            if (result.description.includes("No enrolled") || result.description.includes("denied") || result.description.includes("not found")) {
                const promptUser = lastAttemptedUsername || "NewUser";
                suggestedUserEl.textContent = promptUser;
                btnLabelEl.textContent = promptUser;
                autoRegContainer.style.display = 'block';
                
                // One-click Auto Register action
                document.getElementById('btn-auto-register').onclick = () => autoRegisterUser(promptUser);
            } else {
                autoRegContainer.style.display = 'none';
            }
            
            showScreen('screen-auth-denied');
        }
        
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-rounded">verified_user</span> VERIFY ECG & AUTHENTICATE`;
    }
});

// Register User
document.getElementById('btn-register').addEventListener('click', async () => {
    const usernameInput = document.getElementById('register-username');
    const username = usernameInput.value.trim();
    
    if (!username) {
        showToast('Please enter a username.', 'error');
        return;
    }
    
    if (registerFiles.length === 0) {
        showToast('Please upload ECG recordings (.csv, .txt, or .hea/.dat pair) to extract the biometric signature.', 'error');
        return;
    }
    
    const btn = document.getElementById('btn-register');
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-rounded logo-icon">autorenew</span> REGISTERING PROFILE...`;
    
    const formData = new FormData();
    formData.append('username', username);
    registerFiles.forEach(file => {
        formData.append('files', file);
    });
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/users/register`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Registration failed.');
        }

        // Cache files for instant authentication later
        lastUploadedFiles = [...registerFiles];
        lastAttemptedUsername = username;
        
        // Show success
        document.getElementById('registered-username-label').textContent = result.username;
        showScreen('screen-enrollment-complete');
        
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-rounded">how_to_reg</span> REGISTER USER WITH ECG`;
    }
});

// Auto-register from denial screen
async function autoRegisterUser(username) {
    showToast(`Auto-registering '${username}' with uploaded ECG...`);
    const btn = document.getElementById('btn-auto-register');
    btn.disabled = true;
    
    const formData = new FormData();
    formData.append('username', username);
    lastUploadedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/users/register`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || 'Registration failed.');
        }
        
        document.getElementById('registered-username-label').textContent = result.username;
        showScreen('screen-enrollment-complete');
        showToast('User enrolled successfully!', 'success');
    } catch (e) {
        showToast(e.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

// -------------------------------------------------------------
// DEVELOPER PORTAL LOGIC & POLLING
// -------------------------------------------------------------

// Login
document.getElementById('btn-dev-login').addEventListener('click', async () => {
    const passwordInput = document.getElementById('dev-password');
    const password = passwordInput.value;
    
    if (!password) {
        showToast('Please enter the developer password.', 'error');
        return;
    }
    
    try {
        const formData = new FormData();
        formData.append('password', password);
        
        const response = await fetch(`${BACKEND_URL}/api/dev/login`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Developer login failed.');
        }
        
        devToken = result.token;
        localStorage.setItem('dev_token', devToken);
        passwordInput.value = ''; // clear input
        
        showToast('Developer Access Granted.', 'success');
        showScreen('screen-dev-dashboard');
        
    } catch (error) {
        showToast(error.message, 'error');
    }
});

// Logout
document.getElementById('btn-dev-logout').addEventListener('click', () => {
    devToken = '';
    localStorage.removeItem('dev_token');
    // Set cookie to expire immediately
    document.cookie = "dev_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    showToast('Logged out of Developer Console.');
    showScreen('screen-main');
});

// Refresh Dashboard Button
document.getElementById('btn-refresh-dashboard').addEventListener('click', () => {
    refreshDashboardData();
    showToast('Dashboard details refreshed.');
});

// Init Dashboard
function initDashboard() {
    refreshDashboardData();
    
    // Set 30s polling
    if (dashboardInterval) clearInterval(dashboardInterval);
    dashboardInterval = setInterval(refreshDashboardData, 30000);
}

// Fetch metrics and logs
async function refreshDashboardData() {
    if (!devToken) return;
    
    const activeFilter = document.querySelector('.filter-tabs .tab-btn.active').dataset.filter;
    
    try {
        // 1. Fetch Metrics
        const metricsRes = await fetch(`${BACKEND_URL}/api/metrics/performance`, {
            headers: { 'Authorization': `Bearer ${devToken}` }
        });
        
        if (metricsRes.status === 401) {
            handleDevSessionExpiry();
            return;
        }
        
        const metrics = await metricsRes.json();
        
        // Update stats widgets
        document.getElementById('dashboard-accuracy-badge').textContent = metrics.current_accuracy;
        document.getElementById('stat-replay-buffer').textContent = metrics.replay_buffer_size;
        document.getElementById('stat-enrolled-users').textContent = metrics.enrolled_users;
        
        // Plot TCN-OCL chart
        plotChart(metrics.time_series);
        
        // 2. Fetch Logs
        const logsRes = await fetch(`${BACKEND_URL}/api/logs?status=${activeFilter}`, {
            headers: { 'Authorization': `Bearer ${devToken}` }
        });
        
        const logs = await logsRes.json();
        renderLogsTable(logs);
        
    } catch (error) {
        console.error('Error refreshing dashboard data:', error);
    }
}

function handleDevSessionExpiry() {
    showToast('Developer session expired. Please log in again.', 'error');
    showScreen('screen-dev-login');
}

// Render Logs Table
function renderLogsTable(logs) {
    const tbody = document.getElementById('logs-table-body');
    if (!tbody) return;
    
    if (logs.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="empty-table-cell">No log entries matching filter.</td></tr>`;
        return;
    }
    
    tbody.innerHTML = '';
    logs.forEach(log => {
        const tr = document.createElement('tr');
        
        // Status Badge styling
        let statusBadge = '';
        if (log.status === 'AUTH_APPROVED') {
            statusBadge = `<span class="status-badge badge-approved">AUTH APPROVED</span>`;
        } else if (log.status === 'FAILED') {
            statusBadge = `<span class="status-badge badge-failed">FAILED ATTEMPT</span>`;
        } else if (log.status === 'VERIFICATION_ERROR') {
            statusBadge = `<span class="status-badge badge-error">VERIFICATION ERROR</span>`;
        } else {
            statusBadge = `<span class="status-badge badge-calib">SUCCESS</span>`;
        }
        
        // Timestamp formatting
        const date = new Date(log.created_at);
        const timeStr = date.toLocaleString();
        
        // Accuracy
        const accStr = log.accuracy !== null ? `${(log.accuracy * 100).toFixed(2)}%` : 'N/A';
        
        tr.innerHTML = `
            <td>${statusBadge}</td>
            <td><strong>${log.event_type}</strong></td>
            <td><code>${log.username}</code></td>
            <td>${timeStr}</td>
            <td><strong>${accStr}</strong></td>
            <td>${log.description}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Chart Plotting
function plotChart(timeSeriesData) {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    const labels = timeSeriesData.map(d => d.time);
    const data = timeSeriesData.map(d => d.accuracy);
    
    if (performanceChart) {
        // Update data
        performanceChart.data.labels = labels;
        performanceChart.data.datasets[0].data = data;
        performanceChart.update();
        return;
    }
    
    // Create new chart
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Model Recognition Accuracy',
                data: data,
                borderColor: '#6C47FF',
                backgroundColor: 'rgba(108, 71, 255, 0.05)',
                borderWidth: 3,
                tension: 0.3,
                fill: true,
                pointBackgroundColor: '#6C47FF',
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#FFFFFF',
                pointHoverBorderColor: '#6C47FF',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Accuracy: ${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    min: 80,
                    max: 100,
                    grid: {
                        color: 'rgba(226, 226, 236, 0.5)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        color: '#65647C',
                        font: { family: 'Inter' }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#65647C',
                        font: { family: 'Inter' }
                    }
                }
            }
        }
    });
}

// Tab Filter Change
document.querySelectorAll('.filter-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.filter-tabs .tab-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        refreshDashboardData();
    });
});

// Calibration Trigger
document.getElementById('btn-calibrate-now').addEventListener('click', async () => {
    const btn = document.getElementById('btn-calibrate-now');
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-rounded logo-icon">autorenew</span> CALIBRATING...`;
    showToast('Starting TCN experience replay calibration...');
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/dev/calibrate`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${devToken}` }
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.message || 'Calibration service failed.');
        }
        
        if (result.success) {
            showToast('TCN encoder calibrated successfully using the replay buffer!', 'success');
            refreshDashboardData();
        } else {
            showToast(result.message, 'error');
        }
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-rounded">tune</span> CALIBRATE MODEL NOW`;
    }
});

// Wiping / Flashing Logs
document.getElementById('btn-flash-logs').addEventListener('click', async () => {
    const confirmed = confirm("WARNING: Wiping system logs will wipe ALL database records (users, template embeddings, replay buffer) and reset TCN model weights to baseline.\n\nAre you sure you want to proceed?");
    if (!confirmed) return;
    
    try {
        const response = await fetch(`${BACKEND_URL}/api/logs`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${devToken}` }
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to wipe system data.');
        }
        
        showToast('System reset complete. All profiles and logs wiped.', 'success');
        
        // Reset local templates cache
        lastUploadedFiles = [];
        lastAttemptedUsername = '';
        
        refreshDashboardData();
    } catch (error) {
        showToast(error.message, 'error');
    }
});
