function setupModal() {
    const infoBtn = document.getElementById('infoBtn');
    const closeBtn = document.getElementById('closeBtn');
    const infoModal = document.getElementById('infoModal');

    if (!infoBtn || !closeBtn || !infoModal) {
        console.error('Modal elements not found:', { infoBtn, closeBtn, infoModal });
        return;
    }

    infoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        infoModal.style.display = 'block';
    });

    closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        infoModal.style.display = 'none';
    });

    infoModal.addEventListener('click', (e) => {
        if (e.target === infoModal) {
            infoModal.style.display = 'none';
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupModal);
} else {
    setupModal();
}

const map = L.map('map').setView([20, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap',
    maxZoom: 18,
}).addTo(map);

let activeMarker = null;
let activeCircle = null;

map.on('click', (e) => {
    const { lat, lng } = e.latlng;

    if (activeMarker) map.removeLayer(activeMarker);
    if (activeCircle) map.removeLayer(activeCircle);

    activeMarker = L.circleMarker([lat, lng], {
        radius: 10,
        fillColor: '#0ea5e9',
        color: '#fff',
        weight: 3,
        opacity: 1,
        fillOpacity: 0.9
    }).addTo(map);

    const radius = parseInt(document.getElementById('radius').value);
    activeCircle = L.circle([lat, lng], {
        radius: radius * 1000,
        color: '#0ea5e9',
        fillColor: '#0ea5e9',
        fillOpacity: 0.05,
        weight: 2,
        dashArray: '5, 5'
    }).addTo(map);

    forecast(activeMarker._latlng.wrap().lat, activeMarker._latlng.wrap().lng);

});

async function forecast(lat, lon) {
    const days = document.getElementById('days').value;
    const radius = document.getElementById('radius').value;

    document.getElementById('results').innerHTML =
        '<div class="result-section"><div class="loading"><span class="spinner"></span>Se calculează...</div></div>';

    try {
        const [forecastRes, vulnRes] = await Promise.all([
            fetch(`/forecast?lat=${lat}&lon=${lon}&days=${days}&radius_km=${radius}`),
            fetch(`/vulnerability?lat=${lat}&lon=${lon}&zoom=13`).catch(() => null),
        ]);
        const forecastData = await forecastRes.json();
        const vulnData = vulnRes && vulnRes.ok ? await vulnRes.json() : null;
        renderResults(forecastData, vulnData);
    } catch (e) {
        document.getElementById('results').innerHTML =
            `<div class="result-section"><div class="error">Eroare: ${e.message}</div></div>`;
    }
}

function renderResults(data, vuln) {
    const probs = data.probabilities;
    const feats = data.features;
    const lat = data.location.lat.toFixed(2);
    const lon = data.location.lon.toFixed(2);
    const m5_prob = probs['M_ge_5'] || 0;

    let html = `<div class="result-section">
        <h2>Locație</h2>
        <div class="location-info">
            <strong>${lat}°, ${lon}°</strong><br>
            Fereastra: ${data.days} zile | Rază: ${data.radius_km} km
        </div>
        <div id="mini-map" style="width: 100%; height: 150px; margin-top: 12px; border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.2);"></div>
    </div>`;

    html += `<div class="result-section">
        <h2>Hazard Seismic</h2>`;

    const labels = {
        'M_ge_3': { label: 'M ≥ 3.0', class: 'mag-m3' },
        'M_ge_4': { label: 'M ≥ 4.0', class: 'mag-m4' },
        'M_ge_5': { label: 'M ≥ 5.0', class: 'mag-m5' },
        'M_ge_6': { label: 'M ≥ 6.0', class: 'mag-m6' },
        'M_ge_7': { label: 'M ≥ 7.0', class: 'mag-m7' },
    };

    for (const [key, { label, class: cls }] of Object.entries(labels)) {
        if (probs[key] === undefined) continue;
        const pct = (probs[key] * 100).toFixed(1);
        html += `<div class="magnitude-card ${cls}">
            <span class="mag-label">${label}</span>
            <span class="mag-value">${pct}%</span>
        </div>`;
    }

    html += `<div class="risk-gauge">
        <div class="risk-fill" style="width: ${Math.min(100, m5_prob * 300)}%"></div>
    </div>
    </div>`;

    if (vuln) {
        const exposure = vuln.vulnerability_score;
        const hazard = m5_prob;
        const riskScore = (hazard * exposure).toFixed(1);
        const riskLevel = riskScore < 2 ? 'SCĂZUT' : riskScore < 5 ? 'MODERAT' : 'RIDICAT';

        html += `<div class="result-section">
            <h2>Expunere</h2>
            <div style="text-align: center;">
                <div class="exposure-value">${exposure.toFixed(1)} / 100</div>
                <div style="color: #94a3b8; font-size: 11px; margin-top: 4px;">vulnerabilitate de folosire a terenului</div>
            </div>
        </div>`;

        html += `<div class="result-section">
            <h2>Risc Final</h2>
            <div style="text-align: center;">
                <div class="risk-score">${riskScore}</div>
                <div style="color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">${riskLevel}</div>
            </div>
        </div>`;
    }

    const m3_count = feats.n_m3_20y || 0;
    const m5_count = feats.n_m5_20y || 0;
    const m6_count = feats.n_m6_20y || 0;
    const m7_count = feats.n_m7_20y || 0;

    html += `<div class="result-section">
        <h2>Context Seismic</h2>
        <div class="context-grid">
            <div class="context-item">
                <div class="context-label">M≥3 (20 ani)</div>
                <div class="context-value">${m3_count}</div>
            </div>
            <div class="context-item">
                <div class="context-label">M≥5 (20 ani)</div>
                <div class="context-value">${m5_count}</div>
            </div>
            <div class="context-item">
                <div class="context-label">M≥6 (20 ani)</div>
                <div class="context-value">${m6_count}</div>
            </div>
            <div class="context-item">
                <div class="context-label">M≥7 (20 ani)</div>
                <div class="context-value">${m7_count}</div>
            </div>
        </div>
    </div>`;

    document.getElementById('results').innerHTML = html;

    setTimeout(() => {
        const miniMapEl = document.getElementById('mini-map');
        const radius = parseInt(document.getElementById('radius').value);

        if (miniMapEl._leafletMap) {
            miniMapEl._leafletMap.remove();
            miniMapEl._leafletMap = null;
        }

        const miniMap = L.map(miniMapEl, {
            dragging: false,
            zoomControl: false,
            scrollWheelZoom: false,
            doubleClickZoom: false,
            touchZoom: false,
            keyboard: false
        }).setView([lat, lon], 10);

        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '© Esri',
            maxZoom: 18,
        }).addTo(miniMap);

        const circle = L.circle([lat, lon], {
            radius: radius * 1000,
            color: '#0ea5e9',
            fillColor: '#0ea5e9',
            fillOpacity: 0.1,
            weight: 2.5,
            dashArray: '4, 3'
        }).addTo(miniMap);

        L.circleMarker([lat, lon], {
            radius: 7,
            fillColor: '#fff',
            color: '#0ea5e9',
            weight: 2.5,
            opacity: 1,
            fillOpacity: 1
        }).addTo(miniMap);

        miniMapEl._leafletMap = miniMap;
        miniMap.invalidateSize();

        miniMap.fitBounds(circle.getBounds(), { padding: [30, 30] });
    }, 100);
}

document.getElementById('radius').addEventListener('change', () => {
    if (activeCircle && activeMarker) {
        const lat = activeMarker.getLatLng().lat;
        const lng = activeMarker.getLatLng().lng;
        const radius = parseInt(document.getElementById('radius').value);

        map.removeLayer(activeCircle);
        activeCircle = L.circle([lat, lng], {
            radius: radius * 1000,
            color: '#0ea5e9',
            fillColor: '#0ea5e9',
            fillOpacity: 0.05,
            weight: 2,
            dashArray: '5, 5'
        }).addTo(map);

        forecast(activeMarker._latlng.wrap().lat, activeMarker._latlng.wrap().lng);
    }
});

document.getElementById('days').addEventListener('change', () => {
    if (activeCircle && activeMarker) {
        forecast(activeMarker._latlng.wrap().lat, activeMarker._latlng.wrap().lng);
    }
});
