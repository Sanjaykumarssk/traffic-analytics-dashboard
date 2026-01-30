// Traffic Route Analysis Frontend JavaScript

let map;
let routeLayers = [];
let markers = [];
let currentRouteData = null;
let selectedRoutes = new Set();
let currentRouteId = null; // Store current route ID globally

// Initialize map
function initMap() {
    const mapElement = document.getElementById('map');
    if (!mapElement) {
        console.error('Map element not found');
        return;
    }
    
    // Check if map is already initialized
    if (map) {
        map.remove();
    }
    
    try {
        map = L.map('map', {
            zoomControl: true,
            scrollWheelZoom: true
        }).setView([13.0827, 80.2707], 12); // Default to Chennai

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19,
            tileSize: 256,
            zoomOffset: 0
        }).addTo(map);
        
        // Force map to invalidate size after a short delay
        setTimeout(() => {
            if (map) {
                map.invalidateSize();
            }
        }, 100);
        
        console.log('Map initialized successfully');
    } catch (error) {
        console.error('Error initializing map:', error);
    }
}

// Autocomplete functionality
let originAutocomplete = null;
let destAutocomplete = null;

async function setupAutocomplete(inputId, dropdownId) {
    const input = document.getElementById(inputId);
    const dropdown = document.getElementById(dropdownId);
    
    if (!input || !dropdown) {
        console.error(`Autocomplete setup failed: ${inputId} or ${dropdownId} not found`);
        return;
    }
    
    let timeout;

    // Ensure dropdown has proper initial styling
    dropdown.style.cssText = `
        position: absolute !important;
        z-index: 999999 !important;
        top: 100% !important;
        left: 0 !important;
        right: 0 !important;
        margin-top: 4px !important;
        display: none !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: white !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        max-height: 300px !important;
        overflow-y: auto !important;
        width: 100% !important;
    `;

    input.addEventListener('input', async (e) => {
        let query = e.target.value.trim();
        
        // Clean up query - remove extra commas and spaces
        query = query.replace(/,+/g, ' ').replace(/\s+/g, ' ').trim();

        clearTimeout(timeout);

        if (query.length < 1) {
            dropdown.style.display = 'none';
            dropdown.style.visibility = 'hidden';
            return;
        }

        timeout = setTimeout(async () => {
            try {
                console.log('Fetching autocomplete for:', query);
                
                // Show loading state
                dropdown.innerHTML = '<div style="padding: 12px; text-align: center; color: #718096;">Searching...</div>';
                dropdown.style.cssText = `
                    display: block !important;
                    visibility: visible !important;
                    opacity: 1 !important;
                    z-index: 999999 !important;
                    position: absolute !important;
                    background: white !important;
                    border: 2px solid #e2e8f0 !important;
                    border-radius: 8px !important;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
                    max-height: 300px !important;
                    overflow-y: auto !important;
                    width: 100% !important;
                    margin-top: 4px !important;
                    top: 100% !important;
                    left: 0 !important;
                    right: 0 !important;
                `;
                
                const response = await fetch(`/autocomplete?q=${encodeURIComponent(query)}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Autocomplete response:', data);

                dropdown.innerHTML = '';

                if (data.suggestions && data.suggestions.length > 0) {
                    data.suggestions.forEach((suggestion, index) => {
                        const item = document.createElement('div');
                        item.className = 'autocomplete-item';
                        
                        // Create better display with location details
                        const mainText = suggestion.text || 'Unknown';
                        const address = suggestion.address || {};
                        const city = address.municipality || '';
                        const state = address.countrySubdivision || '';
                        const country = address.country || '';
                        
                        let displayText = mainText;
                        if (city && !mainText.includes(city)) {
                            displayText += `, ${city}`;
                        }
                        if (state && !displayText.includes(state)) {
                            displayText += `, ${state}`;
                        }
                        
                        item.innerHTML = `
                            <div style="font-weight: 600; color: #1a202c;">${mainText}</div>
                            ${city || state ? `<div style="font-size: 12px; color: #718096; margin-top: 2px;">${[city, state, country].filter(Boolean).join(', ')}</div>` : ''}
                        `;
                        
                        item.style.cursor = 'pointer';
                        item.style.padding = '12px 15px';
                        item.style.borderBottom = '1px solid #e2e8f0';
                        item.style.transition = 'background 0.2s';
                        item.style.background = 'white';
                        item.style.color = '#1a202c';
                        
                        item.onmouseenter = () => {
                            item.style.background = '#f7fafc';
                        };
                        item.onmouseleave = () => {
                            item.style.background = 'white';
                        };
                        
                        item.onclick = (e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            
                            // Prevent blur from firing immediately
                            input.focus();
                            
                            // Set the full display text (not just mainText)
                            input.value = displayText;
                            
                            // Store the full location name for route analysis
                            input.dataset.locationName = displayText;
                            input.dataset.mainText = mainText;

                            // Store selected position if available
                            if (suggestion.position) {
                                input.dataset.lat = suggestion.position.lat;
                                input.dataset.lon = suggestion.position.lon;
                            }
                            
                            // Hide dropdown first
                            dropdown.style.display = 'none';
                            dropdown.style.visibility = 'hidden';
                            
                            // Small delay to ensure value is set before blur
                            setTimeout(() => {
                                // Trigger events to ensure value is set
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                                input.dispatchEvent(new Event('change', { bubbles: true }));
                                
                                // Verify value was set correctly
                                if (input.value !== displayText) {
                                    console.warn('Value mismatch, correcting...', input.value, 'should be', displayText);
                                    input.value = displayText;
                                    input.dataset.locationName = displayText;
                                }
                                
                                console.log('Selected location:', displayText, 'for input:', inputId, 'Coordinates:', suggestion.position);
                            }, 50);
                        };
                        dropdown.appendChild(item);
                    });
                    
                    // Force dropdown to be visible with all necessary styles
                    dropdown.style.cssText = `
                        display: block !important;
                        visibility: visible !important;
                        opacity: 1 !important;
                        z-index: 999999 !important;
                        position: absolute !important;
                        background: white !important;
                        border: 2px solid #e2e8f0 !important;
                        border-radius: 8px !important;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
                        max-height: 300px !important;
                        overflow-y: auto !important;
                        width: 100% !important;
                        margin-top: 4px !important;
                        top: 100% !important;
                        left: 0 !important;
                        right: 0 !important;
                    `;
                    
                    // Log for debugging
                    console.log('Dropdown shown with', data.suggestions.length, 'items');
                    
                    // Force reflow
                    void dropdown.offsetHeight;
                } else {
                    dropdown.innerHTML = '<div style="padding: 12px; text-align: center; color: #718096;">No suggestions found</div>';
                    dropdown.style.display = 'block';
                    dropdown.style.visibility = 'visible';
                    // Hide after 2 seconds if no results
                    setTimeout(() => {
                        dropdown.style.display = 'none';
                        dropdown.style.visibility = 'hidden';
                    }, 2000);
                }
            } catch (error) {
                console.error('Autocomplete error:', error);
                dropdown.innerHTML = `<div style="padding: 12px; text-align: center; color: #ef4444;">Error: ${error.message}</div>`;
                dropdown.style.display = 'block';
                dropdown.style.visibility = 'visible';
                // Hide error after 3 seconds
                setTimeout(() => {
                    dropdown.style.display = 'none';
                    dropdown.style.visibility = 'hidden';
                }, 3000);
            }
        }, 200);  // Reduced delay for faster response
    });

    // Hide dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (input && dropdown) {
            if (!input.contains(e.target) && !dropdown.contains(e.target)) {
                dropdown.style.display = 'none';
                dropdown.style.visibility = 'hidden';
            }
        }
    });
    
    // Also hide on input blur (optional) - but allow time for clicks
    let blurTimeout;
    input.addEventListener('blur', (e) => {
        // Clear any existing timeout
        clearTimeout(blurTimeout);
        
        // Longer delay to allow click on dropdown item, especially for destination field
        blurTimeout = setTimeout(() => {
            // Check if dropdown contains the active element (clicked item)
            const activeElement = document.activeElement;
            const dropdownContainsActive = dropdown && dropdown.contains(activeElement);
            
            // Also check if the input value was just set (indicating a selection was made)
            const justSelected = input.dataset.locationName && input.value === input.dataset.locationName;
            
            if (!dropdownContainsActive && !justSelected) {
                dropdown.style.display = 'none';
                dropdown.style.visibility = 'hidden';
            }
        }, 300); // Increased delay for destination field compatibility
    });
    
    // Prevent blur if clicking on dropdown
    dropdown.addEventListener('mousedown', (e) => {
        e.preventDefault(); // Prevent input from losing focus
    });
}

// Analyze route
async function analyzeRoute() {
    console.log('analyzeRoute() called');
    
    try {
        // Prevent admin users from using route analyzer
        const user = await checkAuth();
        if (user && user.is_admin) {
            alert('Route analyzer is not available for admin users. Please use the Admin Panel for system management.');
            window.location.href = '/admin';
            return;
        }
    } catch (authError) {
        console.warn('Auth check failed, continuing anyway:', authError);
        // Continue even if auth check fails - route analyzer should work without auth
    }
    
    const originInput = document.getElementById('origin-input');
    const destInput = document.getElementById('dest-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const errorDiv = document.getElementById('error-message');
    const resultsContainer = document.getElementById('results-container');

    // Validate elements exist
    if (!originInput || !destInput || !analyzeBtn || !errorDiv || !resultsContainer) {
        console.error('Required elements not found:', {
            originInput: !!originInput,
            destInput: !!destInput,
            analyzeBtn: !!analyzeBtn,
            errorDiv: !!errorDiv,
            resultsContainer: !!resultsContainer
        });
        alert('Error: Page elements not loaded. Please refresh the page.');
        return;
    }

    const origin = originInput.value.trim();
    const dest = destInput.value.trim();

    if (!origin || !dest) {
        errorDiv.innerHTML = '<div class="error" style="padding: 15px; background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; color: #ef4444; margin: 10px 0;">Please enter both origin and destination</div>';
        return;
    }
    
    console.log('Analyzing route from:', origin, 'to:', dest);

    errorDiv.innerHTML = '';
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    resultsContainer.innerHTML = '<div class="loading"><div class="loading-spinner"></div><p>Finding best routes...</p></div>';

    // Clear previous routes
    clearMap();
    selectedRoutes.clear();
    updateComparisonPanel();

    try {
        // Prepare request - use stored location name if available, otherwise use input value
        let originName = originInput.dataset.locationName || origin;
        let destName = destInput.dataset.locationName || dest;
        
        const requestBody = {
            origin: originName,
            destination: destName,
            maxAlternatives: 5,
            alpha: 1.0,
            beta: 0.5,
            gamma: 0.001
        };

        // If coordinates available, use them (preferred for accuracy)
        if (originInput.dataset.lat && originInput.dataset.lon) {
            requestBody.origin = {
                name: originName, // Keep the name for display
                lat: parseFloat(originInput.dataset.lat),
                lon: parseFloat(originInput.dataset.lon)
            };
        }
        if (destInput.dataset.lat && destInput.dataset.lon) {
            requestBody.destination = {
                name: destName, // Keep the name for display
                lat: parseFloat(destInput.dataset.lat),
                lon: parseFloat(destInput.dataset.lon)
            };
        }

        // Get auth token if available
        const token = localStorage.getItem('access_token');
        const headers = {
            'Content-Type': 'application/json'
        };
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }

        console.log('Sending route analysis request:', requestBody);
        
        const response = await fetch('/analyze-route', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(requestBody)
        });

        console.log('Response status:', response.status, response.statusText);

        if (!response.ok) {
            let errorDetail = 'Route analysis failed';
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorData.message || JSON.stringify(errorData);
                console.error('API Error response:', errorData);
            } catch (e) {
                // If response is not JSON, get text
                const errorText = await response.text();
                errorDetail = errorText || `HTTP ${response.status}: ${response.statusText}`;
                console.error('API Error (non-JSON):', errorText);
            }
            throw new Error(errorDetail);
        }

        const data = await response.json();
        currentRouteData = data;
        
        // Debug: Log delay calculation for routes
        if (data.analyzed_routes && data.analyzed_routes.length > 0) {
            console.log('=== Delay Calculation Debug ===');
            data.analyzed_routes.forEach((route, idx) => {
                const calculatedDelay = route.travel_time_s && route.no_traffic_s ? 
                    Math.max(0, route.travel_time_s - route.no_traffic_s) : 0;
                console.log(`Route ${idx + 1}: travel=${route.travel_time_s}s, no_traffic=${route.no_traffic_s}s, provided_delay=${route.delay_s}s, calculated_delay=${calculatedDelay}s (${Math.round(calculatedDelay/60)} min)`);
            });
            console.log('=== End Debug ===');
        }

        // Store route ID globally for report access - match backend format
        // Use the location names from the request (which now includes selected dropdown values)
        // Update originName and destName with data from API response if available
        if (data.origin) {
            if (typeof data.origin === 'string') {
                originName = data.origin;
            } else if (data.origin.name) {
                originName = data.origin.name;
            } else if (data.origin.lat && data.origin.lon) {
                originName = `${data.origin.lat},${data.origin.lon}`;
            }
        } else {
            // Fallback to input values if API doesn't return origin
            originName = originInput.dataset.locationName || originInput.value.trim() || 'Origin';
        }

        if (data.destination) {
            if (typeof data.destination === 'string') {
                destName = data.destination;
            } else if (data.destination.name) {
                destName = data.destination.name;
            } else if (data.destination.lat && data.destination.lon) {
                destName = `${data.destination.lat},${data.destination.lon}`;
            }
        } else {
            // Fallback to input values if API doesn't return destination
            destName = destInput.dataset.locationName || destInput.value.trim() || 'Destination';
        }

        currentRouteId = `${originName}→${destName}`;
        console.log('Route ID stored:', currentRouteId);

        displayResults(data);
        drawRoutes(data);

    } catch (error) {
        console.error('Route analysis error:', error);
        const errorMessage = error.message || 'Unknown error occurred';
        errorDiv.innerHTML = `<div class="error" style="padding: 15px; background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; color: #ef4444; margin: 10px 0;">Error: ${errorMessage}</div>`;
        resultsContainer.innerHTML = '<div class="loading" style="padding: 20px; text-align: center; color: #ef4444;">❌ Error analyzing route. Please check the console for details and try again.</div>';
        
        // Log detailed error information
        if (error.stack) {
            console.error('Error stack:', error.stack);
        }
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '🔍 Find Best Routes';
    }
}

// Display results
function displayResults(data) {
    const container = document.getElementById('results-container');
    container.innerHTML = '';

    if (!data.analyzed_routes || data.analyzed_routes.length === 0) {
        container.innerHTML = '<div class="loading">No routes found</div>';
        return;
    }

    // Get origin and destination names for save route function
    let originName = 'Origin';
    let destName = 'Destination';
    if (data.origin) {
        if (typeof data.origin === 'string') {
            originName = data.origin;
        } else if (data.origin.name) {
            originName = data.origin.name;
        }
    }
    if (data.destination) {
        if (typeof data.destination === 'string') {
            destName = data.destination;
        } else if (data.destination.name) {
            destName = data.destination.name;
        }
    }

    // Sort routes by cost (best first)
    const sortedRoutes = [...data.analyzed_routes].sort((a, b) => a.calculated_cost - b.calculated_cost);

    sortedRoutes.forEach((route, displayIdx) => {
        const isBest = route.route_index === data.best_route_index;
        const card = document.createElement('div');
        card.className = `route-card ${isBest ? 'best' : ''}`;
        card.dataset.routeIndex = route.route_index;

        const travelTimeMin = Math.round(route.travel_time_s / 60);
        
        // Always calculate delay from travel_time and no_traffic_time
        // Delay = travel_time - no_traffic_time (the difference is the traffic delay)
        let delayValue = 0;
        if (route.travel_time_s && route.no_traffic_s) {
            // Always calculate from travel_time and no_traffic_time
            delayValue = Math.max(0, route.travel_time_s - route.no_traffic_s);
        } else if (route.delay_s && route.delay_s > 0) {
            // Fallback to provided delay_s only if calculation not possible
            delayValue = route.delay_s;
        }
        
        // Log for debugging - show all values
        console.log(`Route ${route.route_index + 1} Delay Calculation:`, {
            travel_time_s: route.travel_time_s,
            no_traffic_s: route.no_traffic_s,
            provided_delay_s: route.delay_s,
            calculated_delay_s: delayValue,
            delay_min: Math.round(delayValue / 60)
        });
        
        const delayMin = Math.round(delayValue / 60);
        const distanceKm = (route.length_m / 1000).toFixed(1);
        const congestionRatio = route.congestion_ratio ? route.congestion_ratio.toFixed(2) : 'N/A';
        const cost = `₹${route.calculated_cost.toFixed(2)}`;
        const mlPred = route.ml_predicted_congestion ? route.ml_predicted_congestion.toFixed(2) : 'N/A';

        // Use global routeId
        const routeId = currentRouteId || `${originName}→${destName}`;

        card.innerHTML = `
            <div class="route-header">
                <span class="route-title">Route ${route.route_index + 1}</span>
                ${isBest ? '<span class="badge best">⭐ BEST</span>' : ''}
            </div>
            <div class="route-metrics">
                <div class="metric">
                    <div class="metric-label">Travel Time</div>
                    <div class="metric-value">${travelTimeMin} min</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Delay</div>
                    <div class="metric-value">${delayMin} min</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Distance</div>
                    <div class="metric-value">${distanceKm} km</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Congestion</div>
                    <div class="metric-value">${congestionRatio}x</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Cost</div>
                    <div class="metric-value">${cost}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">ML Prediction</div>
                    <div class="metric-value">${mlPred}</div>
                </div>
            </div>
            <div class="checkbox-wrapper" style="display: flex !important; align-items: center; gap: 10px; margin-top: 20px; padding: 15px; background: #e3f2fd; border: 2px solid #667eea; border-radius: 8px; cursor: pointer;" onclick="event.stopPropagation(); (function() { const cb = document.getElementById('route-${route.route_index}'); if (cb) { cb.checked = !cb.checked; toggleRouteSelection(${route.route_index}); } })();">
                <input type="checkbox" id="route-${route.route_index}" 
                       onchange="toggleRouteSelection(${route.route_index});"
                       style="width: 24px !important; height: 24px !important; cursor: pointer !important; accent-color: #667eea !important; flex-shrink: 0; pointer-events: auto;">
                <label for="route-${route.route_index}" style="cursor: pointer; font-weight: 700; color: #1976d2; user-select: none; font-size: 14px; flex: 1; pointer-events: none;">
                    ✓ SELECT THIS ROUTE FOR COMPARISON
                </label>
            </div>
            <div class="route-actions" style="margin-top: 15px; display: flex !important; gap: 10px !important; flex-direction: column;">
                <button class="btn-small btn-view-report" 
                        onclick="console.log('View Report clicked for route:', ${route.route_index}, 'routeId:', '${routeId}'); viewAnalysisReport('${routeId}', ${route.route_index});"
                        style="width: 100% !important; padding: 15px !important; font-size: 15px !important; font-weight: 700 !important; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; cursor: pointer !important; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important; display: block !important; visibility: visible !important;">
                    📊 VIEW DETAILED DASHBOARD REPORT
                </button>
                <button class="btn-small btn-save-route" 
                        onclick="saveRoute('${routeId}', ${route.route_index}, '${originName}', '${destName}');"
                        style="width: 100% !important; padding: 12px !important; font-size: 14px !important; font-weight: 600 !important; background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important; color: white !important; border: none !important; border-radius: 8px !important; cursor: pointer !important; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4) !important; display: block !important; visibility: visible !important;">
                    💾 SAVE THIS ROUTE
                </button>
            </div>
        `;

        // Remove card onclick - selection is handled by checkbox wrapper

        container.appendChild(card);
    });
}

// Toggle route selection
function toggleRouteSelection(routeIndex) {
    console.log('toggleRouteSelection called for route:', routeIndex);
    const checkbox = document.getElementById(`route-${routeIndex}`);
    const card = document.querySelector(`[data-route-index="${routeIndex}"]`);

    if (!checkbox || !card) {
        console.error('Checkbox or card not found for route:', routeIndex);
        return;
    }

    console.log('Checkbox checked state:', checkbox.checked);

    if (checkbox.checked) {
        selectedRoutes.add(routeIndex);
        card.classList.add('selected');
        if (!card.querySelector('.badge.selected')) {
            const badge = document.createElement('span');
            badge.className = 'badge selected';
            badge.textContent = '✓ SELECTED';
            card.querySelector('.route-header').appendChild(badge);
        }
        console.log('Route', routeIndex, 'selected. Total selected:', selectedRoutes.size);
    } else {
        selectedRoutes.delete(routeIndex);
        card.classList.remove('selected');
        const selectedBadge = card.querySelector('.badge.selected');
        if (selectedBadge) {
            selectedBadge.remove();
        }
        console.log('Route', routeIndex, 'deselected. Total selected:', selectedRoutes.size);
    }

    updateComparisonPanel();
    highlightSelectedRoutes();
}

// Update comparison panel
function updateComparisonPanel() {
    const panel = document.getElementById('comparison-panel');
    const tbody = document.getElementById('comparison-tbody');

    if (selectedRoutes.size < 2) {
        panel.classList.remove('active');
        return;
    }

    panel.classList.add('active');
    tbody.innerHTML = '';

    if (!currentRouteData) return;

    const selectedRoutesArray = Array.from(selectedRoutes);
    selectedRoutesArray.forEach((routeIndex, displayIndex) => {
        const route = currentRouteData.analyzed_routes.find(r => r.route_index === routeIndex);
        if (!route) return;

        const row = document.createElement('tr');
        // Add color class based on route index (cycle through colors if more than 10 routes)
        const colorIndex = route.route_index % 11;
        row.className = `route-${colorIndex}`;
        
        row.innerHTML = `
            <td><strong>Route ${route.route_index + 1}</strong></td>
            <td>${Math.round(route.travel_time_s / 60)} min</td>
                            <td>${(() => {
                                let delayVal = route.delay_s || 0;
                                if (!delayVal && route.travel_time_s && route.no_traffic_s) {
                                    delayVal = Math.max(0, route.travel_time_s - route.no_traffic_s);
                                }
                                return Math.round(delayVal / 60);
                            })()} min</td>
            <td>${(route.length_m / 1000).toFixed(1)} km</td>
            <td>${route.congestion_ratio ? route.congestion_ratio.toFixed(2) : 'N/A'}</td>
            <td>₹${route.calculated_cost.toFixed(2)}</td>
            <td>${route.ml_predicted_congestion ? route.ml_predicted_congestion.toFixed(2) : 'N/A'}</td>
        `;
        tbody.appendChild(row);
    });
}

// Highlight selected routes on map
function highlightSelectedRoutes() {
    // Define distinct colors for different routes (same as in drawRoutes)
    const routeColors = [
        '#2196f3', // Blue - Route 0
        '#4caf50', // Green - Route 1
        '#ff9800', // Orange - Route 2
        '#9c27b0', // Purple - Route 3
        '#f44336', // Red - Route 4
        '#00bcd4', // Cyan - Route 5
        '#ffeb3b', // Yellow - Route 6
        '#795548', // Brown - Route 7
        '#607d8b', // Blue Grey - Route 8
        '#e91e63', // Pink - Route 9
        '#3f51b5'  // Indigo - Route 10
    ];
    
    // Update route styles
    routeLayers.forEach((layer, idx) => {
        if (currentRouteData) {
            const route = currentRouteData.analyzed_routes[idx];
            if (route) {
                const isBest = route.route_index === currentRouteData.best_route_index;
                const isSelected = selectedRoutes.has(route.route_index);
                
                // Always use distinct color for each route based on route index
                const color = routeColors[route.route_index % routeColors.length];
                
                // Make selected routes thicker and more opaque, best route slightly thicker
                const weight = isSelected ? 7 : (isBest ? 6 : 5);
                const opacity = isSelected ? 1.0 : (isBest ? 0.95 : 0.8);
                
                layer.setStyle({ color, weight, opacity });
            }
        }
    });
}

// View analysis report
async function viewAnalysisReport(routeId, routeIndex) {
    // Prevent admin users from accessing detailed reports
    const user = await checkAuth();
    if (user && user.is_admin) {
        alert('Detailed reports are not available for admin users. Please use the Admin Panel for system management.');
        window.location.href = '/admin';
        return;
    }
    
    // Ensure we have a valid routeId
    if (!routeId || routeId === 'Origin→Destination') {
        // Try to get from currentRouteData
        if (currentRouteData) {
            const originName = currentRouteData.origin?.name || currentRouteData.origin?.lat || 'Origin';
            const destName = currentRouteData.destination?.name || currentRouteData.destination?.lat || 'Destination';
            routeId = `${originName}→${destName}`;
        } else {
            alert('Please analyze a route first');
            return;
        }
    }

    // Use global routeId if available
    if (currentRouteId) {
        routeId = currentRouteId;
    }

    const url = `/analysis-report?route_id=${encodeURIComponent(routeId)}&route_index=${routeIndex}`;
    console.log('Opening report with routeId:', routeId, 'routeIndex:', routeIndex);
    window.open(url, '_blank');
}

// Draw routes on map
function drawRoutes(data) {
    const origin = data.origin;
    const destination = data.destination;

    // Clear previous
    clearMap();

    // Add markers
    const originPos = [origin.lat || origin.position?.lat, origin.lon || origin.position?.lon];
    const destPos = [destination.lat || destination.position?.lat, destination.lon || destination.position?.lon];

    const originMarker = L.marker(originPos)
        .addTo(map)
        .bindPopup(`<strong>Origin:</strong> ${origin.name || 'Origin'}`);

    const destMarker = L.marker(destPos)
        .addTo(map)
        .bindPopup(`<strong>Destination:</strong> ${destination.name || 'Destination'}`);

    markers.push(originMarker, destMarker);

    // Define distinct colors for different routes
    const routeColors = [
        '#2196f3', // Blue - Route 0
        '#4caf50', // Green - Route 1
        '#ff9800', // Orange - Route 2
        '#9c27b0', // Purple - Route 3
        '#f44336', // Red - Route 4
        '#00bcd4', // Cyan - Route 5
        '#ffeb3b', // Yellow - Route 6
        '#795548', // Brown - Route 7
        '#607d8b', // Blue Grey - Route 8
        '#e91e63', // Pink - Route 9
        '#3f51b5'  // Indigo - Route 10
    ];
    
    // Draw route polylines
    data.analyzed_routes.forEach((route, idx) => {
        const isBest = route.route_index === data.best_route_index;
        const isSelected = selectedRoutes.has(route.route_index);
        
        // Always use distinct color for each route based on route index
        const color = routeColors[route.route_index % routeColors.length];
        
        // Make selected routes thicker and more opaque, best route slightly thicker
        const weight = isSelected ? 7 : (isBest ? 6 : 5);
        const opacity = isSelected ? 1.0 : (isBest ? 0.95 : 0.8);

        if (route.geometry && route.geometry.length > 0) {
            // Draw full geometry
            const polyline = L.polyline(
                route.geometry.map(coord => {
                    if (Array.isArray(coord)) {
                        return [coord[0], coord[1]];
                    }
                    return [coord.lat || coord[0], coord.lon || coord[1]];
                }),
                { color, weight, opacity }
            ).addTo(map);

            // Add popup with route info
            const travelTimeMin = Math.round(route.travel_time_s / 60);
            const distanceKm = (route.length_m / 1000).toFixed(1);
            const bestBadge = isBest ? ' ⭐ BEST' : '';
            const selectedBadge = isSelected ? ' ✓ SELECTED' : '';
            polyline.bindPopup(`
                <strong>Route ${route.route_index + 1}${bestBadge}${selectedBadge}</strong><br>
                Travel Time: ${travelTimeMin} min<br>
                Distance: ${distanceKm} km<br>
                Cost: ₹${route.calculated_cost.toFixed(2)}<br>
                <small style="color: ${color};">● Route Color</small>
            `);

            routeLayers.push(polyline);
        } else {
            // Fallback: draw straight line between origin and destination
            const polyline = L.polyline(
                [originPos, destPos],
                { color, weight, opacity, dashArray: '10, 5' }
            ).addTo(map);
            routeLayers.push(polyline);
        }
    });

    // Fit map to show all routes
    const bounds = L.latLngBounds(originPos, destPos);
    data.analyzed_routes.forEach(route => {
        if (route.geometry && route.geometry.length > 0) {
            route.geometry.forEach(coord => {
                if (Array.isArray(coord)) {
                    bounds.extend([coord[0], coord[1]]);
                } else {
                    bounds.extend([coord.lat || coord[0], coord.lon || coord[1]]);
                }
            });
        }
    });
    map.fitBounds(bounds, { padding: [50, 50] });
}

// Save route function
async function saveRoute(routeId, routeIndex, originName, destName) {
    const token = localStorage.getItem('access_token');
    if (!token) {
        alert('Please login to save routes. Redirecting to login page...');
        window.location.href = '/login';
        return;
    }

    // Get route name from user
    const routeName = prompt(`Enter a name for this route:\n${originName} → ${destName}`, `${originName} → ${destName}`);
    if (!routeName || routeName.trim() === '') {
        return;
    }

    try {
        // Get origin and destination data from currentRouteData
        let originData = {};
        let destData = {};
        
        if (currentRouteData) {
            originData = currentRouteData.origin || { name: originName };
            destData = currentRouteData.destination || { name: destName };
        } else {
            originData = { name: originName };
            destData = { name: destName };
        }

        const response = await fetch('/api/saved-routes', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                route_name: routeName.trim(),
                origin: originData,
                destination: destData,
                route_preferences: {
                    route_index: routeIndex,
                    route_id: routeId
                }
            })
        });

        if (response.ok) {
            alert('✅ Route saved successfully!');
        } else {
            const error = await response.json();
            alert(`Error saving route: ${error.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error saving route:', error);
        alert(`Error saving route: ${error.message}`);
    }
}

// Make saveRoute globally accessible
window.saveRoute = saveRoute;

// Clear map
function clearMap() {
    routeLayers.forEach(layer => map.removeLayer(layer));
    markers.forEach(marker => map.removeLayer(marker));
    routeLayers = [];
    markers = [];
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Initialize map with delay to ensure DOM is ready
    setTimeout(() => {
        initMap();
    }, 100);
    
    // Setup autocomplete - ensure elements exist
    setTimeout(() => {
        const originInput = document.getElementById('origin-input');
        const destInput = document.getElementById('dest-input');
        const originDropdown = document.getElementById('origin-autocomplete');
        const destDropdown = document.getElementById('dest-autocomplete');
        
        if (originInput && originDropdown) {
            console.log('Setting up origin autocomplete');
            setupAutocomplete('origin-input', 'origin-autocomplete');
        } else {
            console.error('Origin autocomplete elements not found');
        }
        
        if (destInput && destDropdown) {
            console.log('Setting up destination autocomplete');
            setupAutocomplete('dest-input', 'dest-autocomplete');
        } else {
            console.error('Destination autocomplete elements not found');
        }
    }, 300);

    // Allow Enter key to trigger analysis
    const originInput = document.getElementById('origin-input');
    const destInput = document.getElementById('dest-input');
    
    if (originInput) {
        originInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                if (typeof analyzeRoute === 'function') {
                    analyzeRoute().catch(error => console.error('Enter key route analysis error:', error));
                } else {
                    console.error('analyzeRoute function not available for Enter key');
                }
            }
        });
    }
    
    if (destInput) {
        destInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                if (typeof analyzeRoute === 'function') {
                    analyzeRoute().catch(error => console.error('Enter key route analysis error:', error));
                } else {
                    console.error('analyzeRoute function not available for Enter key');
                }
            }
        });
    }
    
    // Resize map when window resizes
    window.addEventListener('resize', () => {
        if (map) {
            setTimeout(() => {
                map.invalidateSize();
            }, 100);
        }
    });
});
