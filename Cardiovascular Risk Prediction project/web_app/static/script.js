// Cardiovascular Risk Prediction - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips and form validation
    initializeTooltips();
    setupFormValidation();
    setupEventListeners();
});

function initializeTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(tooltip => {
        tooltip.addEventListener('mouseenter', showTooltip);
        tooltip.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(e) {
    const tooltipText = e.target.getAttribute('data-tooltip');
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = tooltipText;
    document.body.appendChild(tooltip);
    
    const rect = e.target.getBoundingClientRect();
    tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
    tooltip.style.left = rect.left + (rect.width - tooltip.offsetWidth) / 2 + 'px';
}

function hideTooltip() {
    const tooltip = document.querySelector('.tooltip');
    if (tooltip) {
        tooltip.remove();
    }
}

function setupFormValidation() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // Real-time validation
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', validateInput);
    });
}

function validateInput(e) {
    const input = e.target;
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const value = parseFloat(input.value);
    
    if (value < min || value > max) {
        input.classList.add('error');
        showError(input, `Value must be between ${min} and ${max}`);
    } else {
        input.classList.remove('error');
        hideError(input);
    }
}

function showError(input, message) {
    let errorDiv = input.nextElementSibling;
    if (!errorDiv || !errorDiv.classList.contains('error-message')) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        input.parentNode.insertBefore(errorDiv, input.nextSibling);
    }
    errorDiv.textContent = message;
}

function hideError(input) {
    const errorDiv = input.nextElementSibling;
    if (errorDiv && errorDiv.classList.contains('error-message')) {
        errorDiv.remove();
    }
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);
    
    // Validate all fields
    if (!validateForm(data)) {
        return;
    }
    
    // Show loading spinner
    showLoadingSpinner();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        showErrorMessage('An error occurred during prediction. Please try again.');
    } finally {
        hideLoadingSpinner();
    }
}

function validateForm(data) {
    const requiredFields = ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'bmi'];
    
    for (const field of requiredFields) {
        if (!data[field] || data[field] === '') {
            showErrorMessage(`Please fill in ${field.replace('_', ' ')}`);
            return false;
        }
    }
    
    return true;
}

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    if (!resultsDiv) return;
    
    resultsDiv.innerHTML = `
        <div class="result-card">
            <h3>Prediction Results</h3>
            <div class="risk-score ${getRiskClass(result.risk_score)}">
                <span class="score-label">Risk Score:</span>
                <span class="score-value">${(result.risk_score * 100).toFixed(1)}%</span>
            </div>
            <div class="risk-category">
                <span class="category-label">Risk Category:</span>
                <span class="category-value ${getRiskClass(result.risk_score)}">${result.risk_category}</span>
            </div>
            <div class="confidence">
                <span class="confidence-label">Confidence:</span>
                <span class="confidence-value">${(result.confidence * 100).toFixed(1)}%</span>
            </div>
            ${result.recommendations ? generateRecommendations(result.recommendations) : ''}
        </div>
    `;
    
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function getRiskClass(riskScore) {
    if (riskScore < 0.3) return 'low-risk';
    if (riskScore < 0.7) return 'medium-risk';
    return 'high-risk';
}

function generateRecommendations(recommendations) {
    return `
        <div class="recommendations">
            <h4>Recommendations:</h4>
            <ul>
                ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
            </ul>
        </div>
    `;
}

function showLoadingSpinner() {
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    spinner.innerHTML = '<div class="spinner"></div><p>Analyzing data...</p>';
    document.body.appendChild(spinner);
}

function hideLoadingSpinner() {
    const spinner = document.querySelector('.loading-spinner');
    if (spinner) {
        spinner.remove();
    }
}

function showErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

function setupEventListeners() {
    // Reset button
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetForm);
    }
    
    // Export results button
    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportResults);
    }
}

function resetForm() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.reset();
        document.getElementById('results').innerHTML = '';
    }
}

function exportResults() {
    const results = document.getElementById('results');
    if (results && results.innerHTML) {
        const dataStr = JSON.stringify({
            timestamp: new Date().toISOString(),
            results: results.textContent
        });
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `cardiovascular_risk_${Date.now()}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
    }
}

// Chart visualization for results
function createRiskChart(riskScore) {
    const canvas = document.getElementById('risk-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 80;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw background circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.fillStyle = '#f0f0f0';
    ctx.fill();
    
    // Draw risk arc
    const angle = (riskScore * 360 - 90) * Math.PI / 180;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.arc(centerX, centerY, radius, -Math.PI/2, angle);
    ctx.closePath();
    ctx.fillStyle = getRiskColor(riskScore);
    ctx.fill();
    
    // Draw center circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 0.7, 0, 2 * Math.PI);
    ctx.fillStyle = 'white';
    ctx.fill();
    
    // Draw text
    ctx.fillStyle = '#333';
    ctx.font = 'bold 24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${(riskScore * 100).toFixed(0)}%`, centerX, centerY);
}

function getRiskColor(riskScore) {
    if (riskScore < 0.3) return '#4CAF50';
    if (riskScore < 0.7) return '#FFC107';
    return '#F44336';
}
