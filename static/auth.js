// Authentication utilities for frontend

let currentUser = null;

// Check if user is logged in
async function checkAuth() {
    const token = localStorage.getItem('access_token');
    if (!token) {
        return null;
    }

    try {
        const response = await fetch('/api/auth/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            localStorage.removeItem('access_token');
            return null;
        }

        currentUser = await response.json();
        return currentUser;
    } catch (error) {
        console.error('Auth check error:', error);
        localStorage.removeItem('access_token');
        return null;
    }
}

// Logout function
function logout() {
    localStorage.removeItem('access_token');
    currentUser = null;
    window.location.href = '/';
}

// Get current user (synchronous - uses cached value)
function getCurrentUser() {
    return currentUser;
}

// Check if user is admin
function isAdmin() {
    return currentUser && currentUser.is_admin === true;
}

// Initialize auth on page load
async function initAuth() {
    await checkAuth();
    updateUIForAuth();
}

// Update UI based on auth status
function updateUIForAuth() {
    const loginButton = document.getElementById('login-button');
    const userMenu = document.getElementById('user-menu');
    
    if (currentUser) {
        // Hide login button, show user menu
        if (loginButton) loginButton.style.display = 'none';
        if (userMenu) {
            userMenu.style.display = 'flex';
            const usernameSpan = document.getElementById('username-display');
            if (usernameSpan) {
                usernameSpan.textContent = currentUser.is_admin ? '👑 Admin' : currentUser.username;
            }
        }
    } else {
        // Show login button, hide user menu
        if (loginButton) loginButton.style.display = 'block';
        if (userMenu) userMenu.style.display = 'none';
    }
}

// Navigate to user account or admin panel
async function navigateToAccount() {
    // Ensure we have the latest user data
    if (!currentUser) {
        await checkAuth();
    }
    
    if (!currentUser) {
        console.log('No user found, redirecting to login');
        window.location.href = '/login';
        return;
    }
    
    console.log('Navigating to account, user:', currentUser);
    console.log('Is admin?', currentUser.is_admin);
    
    if (currentUser.is_admin) {
        console.log('Redirecting admin to /admin');
        window.location.href = '/admin';
    } else {
        console.log('Redirecting user to /account');
        window.location.href = '/account';
    }
}

