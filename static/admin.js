// Admin Dashboard JavaScript

let currentUser = null;
let currentPage = 0;
let userSkip = 0;
let userLimit = 100;

// Check authentication on page load
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🔐 Admin Panel: Checking authentication...');
    
    // Hide content initially to prevent flash
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.style.display = 'none';
    }
    
    try {
        await checkAuth();
        
        // Check if we were redirected (checkAuth might have redirected)
        if (!localStorage.getItem('access_token')) {
            return; // Already redirected
        }
        
        if (currentUser && currentUser.is_admin) {
            console.log('✅ Admin authenticated:', currentUser.username);
            // Show content after successful auth
            if (mainContent) {
                mainContent.style.display = 'block';
            }
            // Show the admin container
            const adminContainer = document.querySelector('.admin-container');
            if (adminContainer) {
                adminContainer.classList.add('auth-verified');
            }
            initializeAdmin();
        } else {
            console.log('❌ Not admin, redirecting to login...');
            // Redirect to login if not admin
            window.location.href = '/login?redirect=/admin';
        }
    } catch (error) {
        console.error('Auth check failed:', error);
        window.location.href = '/login?redirect=/admin';
    }
    
    // Setup user form handler
    const userForm = document.getElementById('user-form');
    if (userForm) {
        userForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveUser();
        });
    }
});

// Check authentication
async function checkAuth() {
    const token = localStorage.getItem('access_token');
    if (!token) {
        localStorage.removeItem('access_token');
        window.location.replace('/login?redirect=/admin');
        throw new Error('No token');
    }

    try {
        const response = await fetch('/api/auth/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            // If 401 or 403, token is invalid
            if (response.status === 401 || response.status === 403) {
                localStorage.removeItem('access_token');
                window.location.replace('/login?redirect=/admin');
                throw new Error('Not authenticated');
            }
            throw new Error(`Auth check failed: ${response.status}`);
        }

        currentUser = await response.json();
        
        // Verify user is admin
        if (!currentUser || !currentUser.is_admin) {
            localStorage.removeItem('access_token');
            window.location.replace('/login?redirect=/admin');
            throw new Error('Not an admin user');
        }
        
        // Update UI with username
        const usernameElement = document.getElementById('admin-username');
        if (usernameElement) {
            usernameElement.textContent = currentUser.username || 'Admin';
        }
        
        return currentUser;
    } catch (error) {
        console.error('Auth error:', error);
        localStorage.removeItem('access_token');
        // Only redirect if not already redirected
        if (window.location.pathname === '/admin') {
            window.location.replace('/login?redirect=/admin');
        }
        throw error;
    }
}

// Initialize admin dashboard
function initializeAdmin() {
    console.log('🚀 Initializing Admin Dashboard...');
    console.log('Setting up navigation...');
    setupNavigation();
    
    // Verify Access Control section exists
    const accessControlSection = document.getElementById('section-access-control');
    if (accessControlSection) {
        console.log('✅ Access Control section found:', accessControlSection);
    } else {
        console.error('❌ Access Control section NOT FOUND!');
    }
    
    // List all sections
    const allSections = document.querySelectorAll('.content-section');
    console.log('📋 All available sections:', Array.from(allSections).map(s => s.id));
    
    // Setup route filter handler
    const routeFilter = document.getElementById('route-filter');
    if (routeFilter) {
        routeFilter.addEventListener('change', () => {
            console.log('Route filter changed:', routeFilter.value);
            loadRoutes();
        });
    }
    
    loadDashboard();
    loadUsers();
    loadRoutes();
    loadCacheStats();
    checkSystemStatus();
    console.log('✅ Admin Dashboard initialized');
}

// Setup navigation
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    console.log('🔗 Found navigation items:', navItems.length);
    
    navItems.forEach(item => {
        const sectionName = item.dataset.section;
        console.log('  - Navigation item:', sectionName, item.textContent.trim());
        
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const section = item.dataset.section;
            console.log('🖱️ Navigation clicked:', section);
            showSection(section);
            
            // Update active state
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
        });
    });
    
    // Verify Access Control nav item exists
    const accessControlNav = document.querySelector('[data-section="access-control"]');
    if (accessControlNav) {
        console.log('✅ Access Control navigation item found');
    } else {
        console.error('❌ Access Control navigation item NOT FOUND!');
    }
}

// Show section
function showSection(sectionName) {
    console.log('Switching to section:', sectionName);
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.classList.remove('active');
        section.style.display = 'none';
    });
    
    const targetSection = document.getElementById(`section-${sectionName}`);
    if (targetSection) {
        targetSection.classList.add('active');
        targetSection.style.display = 'block';
        console.log('Section activated:', sectionName, targetSection);
    } else {
        console.error('Section not found:', `section-${sectionName}`);
        // List all available sections for debugging
        const allSections = document.querySelectorAll('.content-section');
        console.log('Available sections:', Array.from(allSections).map(s => s.id));
    }

    // Update page title
    const titles = {
        'dashboard': 'Dashboard Overview',
        'users': 'User Management',
        'routes': 'Route Analytics',
        'system': 'System Settings',
        'access-control': 'Admin Access Control',
        'cache': 'Cache Management',
        'reports': 'Reports & Export'
    };
    const pageTitle = document.getElementById('page-title');
    if (pageTitle) {
        pageTitle.textContent = titles[sectionName] || 'Admin Panel';
    }
    
    // Reload routes when routes section is shown
    if (sectionName === 'routes') {
        console.log('[showSection] Routes section shown, loading routes...');
        // Small delay to ensure DOM elements are ready
        setTimeout(() => {
            console.log('[showSection] Calling loadRoutes()...');
            loadRoutes();
        }, 100);
    }
    
    // Load admin settings when system or access-control section is shown
    if (sectionName === 'system' || sectionName === 'access-control') {
        setTimeout(() => {
            if (typeof loadAdminSettings === 'function') {
                console.log('Loading admin settings...');
                loadAdminSettings();
            } else {
                console.warn('loadAdminSettings function not found');
            }
        }, 100);
    }
}

// Load dashboard data
async function loadDashboard() {
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch('/api/admin/stats', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) throw new Error('Failed to load stats');

        const stats = await response.json();
        
        document.getElementById('stat-users').textContent = stats.total_users || 0;
        document.getElementById('stat-routes').textContent = stats.total_route_analyses || 0;
        document.getElementById('stat-saved').textContent = stats.total_saved_routes || 0;
        document.getElementById('stat-ratings').textContent = stats.total_ratings || 0;

        // Load charts
        loadUserGrowthChart();
        loadRouteTrendsChart();
        loadRecentActivity();
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showToast('Error loading dashboard data', 'error');
    }
}

// Load users
async function loadUsers(skip = 0) {
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch(`/api/admin/users?skip=${skip}&limit=${userLimit}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) throw new Error('Failed to load users');

        const users = await response.json();
        const tbody = document.getElementById('users-table-body');
        tbody.innerHTML = '';

        if (users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9">No users found</td></tr>';
            return;
        }

        users.forEach(user => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${user.id}</td>
                <td>${user.username}</td>
                <td>${user.email}</td>
                <td>${user.full_name || '-'}</td>
                <td><span class="status-badge ${user.is_active ? 'active' : 'inactive'}">${user.is_active ? 'Active' : 'Inactive'}</span></td>
                <td><span class="status-badge ${user.is_admin ? 'admin' : 'user'}">${user.is_admin ? 'Admin' : 'User'}</span></td>
                <td>${user.created_at ? new Date(user.created_at).toLocaleDateString() : '-'}</td>
                <td>${user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}</td>
                <td>
                    <div class="action-buttons">
                        <button class="btn-small btn-primary" onclick="editUser(${user.id})" title="Edit user">Edit</button>
                        <button class="btn-small btn-secondary" onclick="toggleUserStatus(${user.id}, ${user.is_active})" title="${user.is_active ? 'Deactivate' : 'Activate'} user">${user.is_active ? 'Deactivate' : 'Activate'}</button>
                        ${!user.is_admin ? `<button class="btn-small btn-secondary" onclick="toggleAdminStatus(${user.id}, ${user.is_admin})" title="${user.is_admin ? 'Revoke admin' : 'Make admin'}">${user.is_admin ? 'Revoke' : 'Make Admin'}</button>` : ''}
                        <button class="btn-small btn-danger" onclick="deleteUser(${user.id})" title="Delete user" style="background: #ef4444; color: white;">Delete</button>
                    </div>
                </td>
            `;
            tbody.appendChild(row);
        });

        userSkip = skip;
        document.getElementById('page-info').textContent = `Page ${Math.floor(skip / userLimit) + 1}`;
    } catch (error) {
        console.error('Error loading users:', error);
        showToast('Error loading users', 'error');
    }
}

// Load routes
async function loadRoutes() {
    console.log('[loadRoutes] Function called');
    console.log('[loadRoutes] Timestamp:', new Date().toISOString());
    
    const tbody = document.getElementById('routes-table-body');
    if (!tbody) {
        console.error('[loadRoutes] routes-table-body element not found in DOM!');
        return;
    }
    
    console.log('[loadRoutes] Setting loading state...');
    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 20px;">Loading...</td></tr>';
    
    try {
        const token = localStorage.getItem('access_token');
        if (!token) {
            console.error('No access token found');
            if (tbody) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 20px; color: #ef4444;">Authentication required. Please login.</td></tr>';
            }
            return;
        }
        
        const filterValue = document.getElementById('route-filter')?.value || 'all';
        console.log('Filter value:', filterValue);
        
        // Build URL with filter
        let url = '/api/admin/route-analysis';
        if (filterValue && filterValue !== 'all') {
            url += `?filter=${filterValue}`;
        }
        
        console.log('Fetching from:', url);
        
        const response = await fetch(url, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
            let errorText = '';
            try {
                errorText = await response.text();
                console.error('Error response:', errorText);
                // Try to parse as JSON for better error message
                try {
                    const errorJson = JSON.parse(errorText);
                    errorText = errorJson.detail || errorJson.message || errorText;
                } catch {
                    // Keep original text if not JSON
                }
            } catch (e) {
                errorText = `HTTP ${response.status}: ${response.statusText}`;
            }
            
            // Show specific error messages
            if (response.status === 401 || response.status === 403) {
                throw new Error('Authentication failed. Please login again.');
            } else if (response.status === 404) {
                throw new Error('Endpoint not found. Server may need to be restarted.');
            } else {
                throw new Error(`Failed to load routes: ${response.status} - ${errorText}`);
                }
        }

        const responseText = await response.text();
        console.log('Response text:', responseText);
        
        let data;
        try {
            data = JSON.parse(responseText);
            console.log('Route data received:', data);
        } catch (parseError) {
            console.error('Failed to parse JSON response:', parseError);
            console.error('Response was:', responseText);
            throw new Error(`Invalid JSON response: ${parseError.message}`);
        }
        
        // Validate data structure
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid data format received from server');
        }
        
        // Update statistics cards
        if (data.stats) {
            const routeTotalEl = document.getElementById('route-total');
            const routeAvgTimeEl = document.getElementById('route-avg-time');
            const routeAvgDelayEl = document.getElementById('route-avg-delay');
            const routeAvgCostEl = document.getElementById('route-avg-cost');
            
            if (routeTotalEl) routeTotalEl.textContent = data.stats.total || 0;
            
            // Convert travel time from seconds to minutes
            if (routeAvgTimeEl) {
                const avgTravelTimeMin = data.stats.avg_travel_time ? (data.stats.avg_travel_time / 60).toFixed(1) : '0';
                routeAvgTimeEl.textContent = avgTravelTimeMin;
            }
            
            // Convert delay from seconds to minutes
            if (routeAvgDelayEl) {
                const avgDelayMin = data.stats.avg_delay ? (data.stats.avg_delay / 60).toFixed(1) : '0';
                routeAvgDelayEl.textContent = avgDelayMin;
            }
            
            // Display average cost
            if (routeAvgCostEl) {
                const avgCost = data.stats.avg_cost ? data.stats.avg_cost.toFixed(2) : '0';
                routeAvgCostEl.textContent = avgCost;
            }
        }
        
        // Update table
        const tbody = document.getElementById('routes-table-body');
        if (!tbody) {
            console.error('routes-table-body element not found!');
            return;
        }

        if (!data.routes || !Array.isArray(data.routes) || data.routes.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 20px;">No route analyses found</td></tr>';
            return;
        }
        
        tbody.innerHTML = '';
        
        data.routes.forEach((route, index) => {
            try {
                const row = document.createElement('tr');
                
                // Format travel time (seconds to minutes)
                const travelTimeMin = route.travel_time_s ? (route.travel_time_s / 60).toFixed(1) : '0';
                
                // Format delay (seconds to minutes) - calculate if not provided or zero
                let delaySeconds = route.delay_s;
                if (!delaySeconds || delaySeconds === 0) {
                    // Calculate delay: delay = travel_time - no_traffic_time
                    if (route.travel_time_s && route.no_traffic_s) {
                        delaySeconds = Math.max(0, route.travel_time_s - route.no_traffic_s);
                    } else {
                        delaySeconds = 0;
                    }
                }
                const delayMin = delaySeconds ? (delaySeconds / 60).toFixed(1) : '0';
                
                // Format distance (meters to kilometers)
                const distanceKm = route.length_m ? (route.length_m / 1000).toFixed(2) : '0';
                
                // Format cost
                const cost = route.calculated_cost ? route.calculated_cost.toFixed(2) : '0';
                
                // Format timestamp
                let timestamp = '-';
                if (route.timestamp) {
                    try {
                        const date = new Date(route.timestamp);
                        timestamp = date.toLocaleString();
                    } catch (e) {
                        timestamp = route.timestamp;
                    }
                }
                
                // Escape route_id for onclick
                const routeIdEscaped = (route.route_id || '').replace(/'/g, "\\'");
                const routeNameEscaped = (route.route || '-').replace(/"/g, '&quot;');
                
                row.innerHTML = `
                    <td>${route.id || index + 1}</td>
                    <td>${routeNameEscaped}</td>
                    <td>${travelTimeMin} min</td>
                    <td>${delayMin} min</td>
                    <td>${distanceKm} km</td>
                    <td>${cost}</td>
                    <td>${timestamp}</td>
                    <td>
                        <div class="action-buttons">
                            <button class="btn-small btn-primary" onclick="viewRouteDetails(${route.id || 0}, '${routeIdEscaped}')" title="View route details">View</button>
                        </div>
                    </td>
                `;
                tbody.appendChild(row);
            } catch (rowError) {
                console.error(`Error creating row for route ${index}:`, rowError, route);
            }
        });
        
        console.log(`Successfully loaded ${data.routes.length} routes`);
        
    } catch (error) {
        console.error('Error loading routes:', error);
        const tbody = document.getElementById('routes-table-body');
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 20px; color: #ef4444;">Error loading routes: ${error.message}. Please try again.</td></tr>`;
        }
        showToast('Error loading route analytics: ' + error.message, 'error');
    }
}

// View route details
window.viewRouteDetails = function(routeId, routeIdString) {
    // Extract origin and destination from route_id
    const parts = routeIdString.split('_route')[0];
    const routeParts = parts.split('→');
    if (routeParts.length === 2) {
        const origin = routeParts[0].trim();
        const dest = routeParts[1].trim();
        window.location.href = `/analysis-report?routeId=${encodeURIComponent(parts)}`;
    } else {
        showToast('Unable to parse route information', 'error');
    }
};

// Load cache stats
async function loadCacheStats() {
    try {
        const response = await fetch('/api/cache/stats');
        if (!response.ok) return;

        const stats = await response.json();
        document.getElementById('cache-size').textContent = stats.size || 0;
        document.getElementById('cache-hits').textContent = stats.hits || 0;
        document.getElementById('cache-misses').textContent = stats.misses || 0;
    } catch (error) {
        console.error('Error loading cache stats:', error);
    }
}

// Check system status
async function checkSystemStatus() {
    // Check database
    try {
        const response = await fetch('/health');
        if (response.ok) {
            document.getElementById('db-status').textContent = 'Connected';
            document.getElementById('db-status').className = 'status-badge active';
        } else {
            throw new Error('Database not connected');
        }
    } catch (error) {
        document.getElementById('db-status').textContent = 'Disconnected';
        document.getElementById('db-status').className = 'status-badge inactive';
    }

    // Check ML model
    try {
        const response = await fetch('/health');
        if (response.ok) {
            const data = await response.json();
            if (data.model_loaded) {
                document.getElementById('ml-status').textContent = 'Loaded';
                document.getElementById('ml-status').className = 'status-badge active';
            } else {
                document.getElementById('ml-status').textContent = 'Not Loaded';
                document.getElementById('ml-status').className = 'status-badge inactive';
            }
        }
    } catch (error) {
        document.getElementById('ml-status').textContent = 'Error';
        document.getElementById('ml-status').className = 'status-badge inactive';
    }

    // Check API
    try {
        const response = await fetch('/health');
        if (response.ok) {
            document.getElementById('api-status').textContent = 'Online';
            document.getElementById('api-status').className = 'status-badge active';
        }
    } catch (error) {
        document.getElementById('api-status').textContent = 'Offline';
        document.getElementById('api-status').className = 'status-badge inactive';
    }
}

// User management functions
function showAddUserModal() {
    document.getElementById('modal-title').textContent = 'Add User';
    document.getElementById('user-id').value = '';
    document.getElementById('modal-username').value = '';
    document.getElementById('modal-email').value = '';
    document.getElementById('modal-fullname').value = '';
    document.getElementById('modal-password').value = '';
    document.getElementById('modal-active').checked = true;
    document.getElementById('modal-admin').checked = false;
    document.getElementById('user-modal').style.display = 'block';
}


// Make functions globally accessible
window.toggleUserStatus = async function(userId, currentStatus) {
    if (!confirm(`Are you sure you want to ${currentStatus ? 'deactivate' : 'activate'} this user?`)) return;
    
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch(`/api/admin/users/${userId}/activate`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to update user status' }));
            throw new Error(errorData.detail || 'Failed to update user status');
        }
        
        showToast('User status updated', 'success');
        loadUsers(userSkip);
    } catch (error) {
        console.error('Error updating user status:', error);
        showToast(error.message || 'Error updating user status', 'error');
    }
};

window.toggleAdminStatus = async function(userId, currentStatus) {
    if (!confirm(`Are you sure you want to ${currentStatus ? 'revoke' : 'grant'} admin privileges?`)) return;
    
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch(`/api/admin/users/${userId}/admin`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to update admin status' }));
            throw new Error(errorData.detail || 'Failed to update admin status');
        }
        
        showToast('Admin status updated', 'success');
        loadUsers(userSkip);
    } catch (error) {
        console.error('Error updating admin status:', error);
        showToast(error.message || 'Error updating admin status', 'error');
    }
};

window.deleteUser = async function(userId) {
    if (!confirm('Are you sure you want to delete this user? This action cannot be undone.')) return;
    
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch(`/api/admin/users/${userId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to delete user' }));
            throw new Error(errorData.detail || 'Failed to delete user');
        }
        
        showToast('User deleted successfully', 'success');
        loadUsers(userSkip);
    } catch (error) {
        console.error('Error deleting user:', error);
        showToast(error.message || 'Error deleting user', 'error');
    }
};

window.editUser = function(userId) {
    // Load user data and show in modal
    const token = localStorage.getItem('access_token');
    fetch(`/api/admin/users?skip=0&limit=1000`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    })
    .then(res => res.json())
    .then(users => {
        const user = users.find(u => u.id === userId);
        if (user) {
            document.getElementById('user-id').value = user.id;
            document.getElementById('modal-username').value = user.username;
            document.getElementById('modal-email').value = user.email;
            document.getElementById('modal-fullname').value = user.full_name || '';
            document.getElementById('modal-password').value = ''; // Clear password field
            
            // Set checkbox states - ensure they're enabled and clickable
            const activeCheckbox = document.getElementById('modal-active');
            const adminCheckbox = document.getElementById('modal-admin');
            const selfAdminWarning = document.getElementById('self-admin-warning');
            
            activeCheckbox.disabled = false;
            activeCheckbox.checked = user.is_active !== undefined ? user.is_active : true;
            
            // Check if editing self
            const currentUserId = currentUser ? currentUser.id : null;
            const editingSelf = currentUserId && parseInt(userId) === currentUserId;
            
            adminCheckbox.disabled = false;
            adminCheckbox.checked = user.is_admin !== undefined ? user.is_admin : false;
            
            // Show warning if editing self and they're admin
            if (editingSelf && user.is_admin) {
                if (selfAdminWarning) {
                    selfAdminWarning.style.display = 'block';
                }
            } else {
                if (selfAdminWarning) {
                    selfAdminWarning.style.display = 'none';
                }
            }
            
            document.getElementById('modal-title').textContent = 'Edit User';
            document.getElementById('user-modal').style.display = 'block';
            
            console.log('Edit modal opened - Active:', activeCheckbox.checked, 'Admin:', adminCheckbox.checked, 'Editing self:', editingSelf);
        }
    })
    .catch(err => {
        console.error('Error loading user:', err);
        showToast('Error loading user data', 'error');
    });
};

function closeUserModal() {
    document.getElementById('user-modal').style.display = 'none';
    // Reset form
    const form = document.getElementById('user-form');
    if (form) {
        form.reset();
        document.getElementById('user-id').value = '';
        document.getElementById('modal-password').value = '';
    }
}

// Save user (create or update)
async function saveUser() {
    const userId = document.getElementById('user-id').value;
    const username = document.getElementById('modal-username').value.trim();
    const email = document.getElementById('modal-email').value.trim();
    const fullName = document.getElementById('modal-fullname').value.trim();
    const password = document.getElementById('modal-password').value;
    // Get checkbox values - explicitly read as boolean
    const activeCheckbox = document.getElementById('modal-active');
    const adminCheckbox = document.getElementById('modal-admin');
    const isActive = activeCheckbox ? activeCheckbox.checked : true;
    const isAdmin = adminCheckbox ? adminCheckbox.checked : false;
    
    console.log('Saving user - Checkbox states:');
    console.log('  Is Active:', isActive, '(type:', typeof isActive, ')');
    console.log('  Is Admin:', isAdmin, '(type:', typeof isAdmin, ')');
    
    // Validation
    if (!username) {
        showToast('Username is required', 'error');
        document.getElementById('modal-username').focus();
        return;
    }
    if (!email) {
        showToast('Email is required', 'error');
        document.getElementById('modal-email').focus();
        return;
    }
    if (password && password.length < 8) {
        showToast('Password must be at least 8 characters', 'error');
        document.getElementById('modal-password').focus();
        return;
    }
    
    try {
        const token = localStorage.getItem('access_token');
        if (!token) {
            showToast('Authentication required', 'error');
            return;
        }
        
        // Prepare update data
        const updateData = {
            username: username,
            email: email,
            full_name: fullName || null,
            is_active: isActive,
            is_admin: isAdmin
        };
        
        // Only include password if provided (not empty)
        if (password && password.trim().length > 0) {
            updateData.password = password.trim();
        }
        
        // Only update if user ID exists (edit mode)
        if (!userId) {
            showToast('User ID is required', 'error');
            return;
        }
        
        console.log('Updating user:', userId, updateData);
        console.log('Checkbox states - is_active:', isActive, 'is_admin:', isAdmin);
        
        const response = await fetch(`/api/admin/users/${userId}`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updateData)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to update user' }));
            const errorMessage = errorData.detail || `HTTP ${response.status}: ${response.statusText}`;
            console.error('Update failed:', errorMessage);
            throw new Error(errorMessage);
        }
        
        const updatedUser = await response.json();
        console.log('User updated successfully:', updatedUser);
        
        let successMessage = 'User updated successfully';
        if (!isActive) {
            successMessage += ' - User has been deactivated';
        }
        if (!isAdmin && updateData.is_admin === false) {
            successMessage += ' - Admin privileges removed';
        }
        
        showToast(successMessage, 'success');
        closeUserModal();
        
        // Reload users list to show updated data
        loadUsers(userSkip);
        
    } catch (error) {
        console.error('Error saving user:', error);
        showToast(error.message || 'Failed to update user', 'error');
    }
}

// Make saveUser globally accessible
window.saveUser = saveUser;

// Cache management
async function clearAllCache() {
    if (!confirm('Are you sure you want to clear all cache?')) return;

    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch('/api/cache/clear', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) throw new Error('Failed to clear cache');

        showToast('Cache cleared successfully', 'success');
        loadCacheStats();
    } catch (error) {
        console.error('Error clearing cache:', error);
        showToast('Error clearing cache', 'error');
    }
}

// Charts
function loadUserGrowthChart() {
    // Placeholder for user growth chart
    const ctx = document.getElementById('userGrowthChart');
    if (!ctx) return;

    // Simple chart implementation
    // In production, use Chart.js with real data
}

function loadRouteTrendsChart() {
    // Placeholder for route trends chart
    const ctx = document.getElementById('routeTrendsChart');
    if (!ctx) return;
}

function loadRecentActivity() {
    const activityList = document.getElementById('activity-list');
    activityList.innerHTML = '<div class="activity-item">Recent activity will be displayed here</div>';
}

// Export functions
async function exportUsers() {
    try {
        const token = localStorage.getItem('access_token');
        if (!token) {
            showToast('Please login first', 'error');
            return;
        }

        showToast('Exporting users data...', 'info');
        
        const response = await fetch('/api/admin/export/users/csv', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'text/csv'
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to export users');
        }

        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'users_export.csv';
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Get the CSV content
        const blob = await response.blob();
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showToast('Users data exported successfully! File saved to project/exports folder and downloaded.', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showToast(error.message || 'Failed to export users data', 'error');
    }
}

async function exportRoutes() {
    try {
        const token = localStorage.getItem('access_token');
        if (!token) {
            showToast('Please login first', 'error');
            return;
        }

        showToast('Exporting routes data...', 'info');
        
        const response = await fetch('/api/admin/export/routes/csv', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'text/csv'
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to export routes');
        }

        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'routes_export.csv';
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Get the CSV content
        const blob = await response.blob();
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showToast('Routes data exported successfully! File saved to project/exports folder and downloaded.', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showToast(error.message || 'Failed to export routes data', 'error');
    }
}

async function exportSystem() {
    try {
        const token = localStorage.getItem('access_token');
        if (!token) {
            showToast('Please login first', 'error');
            return;
        }

        showToast('Exporting system report...', 'info');
        
        const response = await fetch('/api/admin/export/system/json', {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to export system report');
        }

        // Get filename from Content-Disposition header or use default
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'system_report.json';
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Get the JSON content
        const blob = await response.blob();
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        
        // Cleanup
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showToast('System report exported successfully! File saved to project/exports folder and downloaded.', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showToast(error.message || 'Failed to export system report', 'error');
    }
}

// System check functions
function checkDatabase() {
    checkSystemStatus();
    showToast('Database status checked', 'info');
}

function checkMLModel() {
    checkSystemStatus();
    showToast('ML model status checked', 'info');
}

function checkAPI() {
    checkSystemStatus();
    showToast('API status checked', 'info');
}

// Refresh data
function refreshData() {
    console.log('Refreshing all data...');
    loadDashboard();
    loadUsers(userSkip);
    loadRoutes(); // Reload routes when refreshing
    loadCacheStats();
    checkSystemStatus();
    showToast('Data refreshed', 'success');
}


// Logout
function logout() {
    localStorage.removeItem('access_token');
    window.location.href = '/login';
}

// Toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';

    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}

// Admin Access Control Functions
function saveAccessControlSettings() {
    const settings = {
        requireAdminApproval: document.getElementById('require-admin-approval').checked,
        logAdminActions: document.getElementById('log-admin-actions').checked,
        restrictUserDeletion: document.getElementById('restrict-user-deletion').checked
    };
    localStorage.setItem('adminAccessControl', JSON.stringify(settings));
    showToast('Access control settings saved', 'success');
}

function savePermissionSettings() {
    const permissions = {
        canManageUsers: document.getElementById('can-manage-users').checked,
        canManageRoutes: document.getElementById('can-manage-routes').checked,
        canClearCache: document.getElementById('can-clear-cache').checked,
        canExportData: document.getElementById('can-export-data').checked
    };
    localStorage.setItem('adminPermissions', JSON.stringify(permissions));
    showToast('Permission settings saved', 'success');
}

function saveSecuritySettings() {
    const security = {
        sessionTimeout: parseInt(document.getElementById('session-timeout').value) || 30,
        require2FA: document.getElementById('require-2fa').checked,
        ipWhitelist: document.getElementById('ip-whitelist').checked
    };
    localStorage.setItem('adminSecurity', JSON.stringify(security));
    showToast('Security settings saved', 'success');
}

// Load admin settings
function loadAdminSettings() {
    // Load access control
    const accessControl = localStorage.getItem('adminAccessControl');
    if (accessControl) {
        try {
            const settings = JSON.parse(accessControl);
            if (document.getElementById('require-admin-approval')) {
                document.getElementById('require-admin-approval').checked = settings.requireAdminApproval || false;
            }
            if (document.getElementById('log-admin-actions')) {
                document.getElementById('log-admin-actions').checked = settings.logAdminActions !== false;
            }
            if (document.getElementById('restrict-user-deletion')) {
                document.getElementById('restrict-user-deletion').checked = settings.restrictUserDeletion !== false;
            }
        } catch (e) {
            console.error('Error loading access control settings:', e);
        }
    }
    
    // Load permissions
    const permissions = localStorage.getItem('adminPermissions');
    if (permissions) {
        try {
            const perms = JSON.parse(permissions);
            if (document.getElementById('can-manage-users')) {
                document.getElementById('can-manage-users').checked = perms.canManageUsers !== false;
            }
            if (document.getElementById('can-manage-routes')) {
                document.getElementById('can-manage-routes').checked = perms.canManageRoutes !== false;
            }
            if (document.getElementById('can-clear-cache')) {
                document.getElementById('can-clear-cache').checked = perms.canClearCache !== false;
            }
            if (document.getElementById('can-export-data')) {
                document.getElementById('can-export-data').checked = perms.canExportData !== false;
            }
        } catch (e) {
            console.error('Error loading permission settings:', e);
        }
    }
    
    // Load security
    const security = localStorage.getItem('adminSecurity');
    if (security) {
        try {
            const sec = JSON.parse(security);
            if (document.getElementById('session-timeout')) {
                document.getElementById('session-timeout').value = sec.sessionTimeout || 30;
            }
            if (document.getElementById('require-2fa')) {
                document.getElementById('require-2fa').checked = sec.require2FA || false;
            }
            if (document.getElementById('ip-whitelist')) {
                document.getElementById('ip-whitelist').checked = sec.ipWhitelist || false;
            }
        } catch (e) {
            console.error('Error loading security settings:', e);
        }
    }
}

// Make functions globally accessible
window.saveAccessControlSettings = saveAccessControlSettings;
window.savePermissionSettings = savePermissionSettings;
window.saveSecuritySettings = saveSecuritySettings;
window.loadAdminSettings = loadAdminSettings;

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('user-modal');
    if (event.target === modal) {
        closeUserModal();
    }
}

