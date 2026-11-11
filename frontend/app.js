const form = document.getElementById('calcForm');
const resultDiv = document.getElementById('result');
const btnPerfilBasico = document.getElementById('btnPerfilBasico');
const btnPerfilTrabalho = document.getElementById('btnPerfilTrabalho');
const btnPerfilGamer = document.getElementById('btnPerfilGamer');
const btnToggleDataset = document.getElementById('btnToggleDataset');
const datasetDrawer = document.getElementById('datasetDrawer');
const datasetTbody = document.getElementById('datasetTbody');
const datasetMeta = document.getElementById('datasetMeta');
const linkCsvRaw = document.getElementById('linkCsvRaw');
const linkTrain = document.getElementById('linkTrain');

// Abre e fecha os balões de ajuda ("?") de cada campo
document.querySelectorAll('.help-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-help-target');
        const targetBox = document.getElementById(targetId);
        if (!targetBox) return;

        const isOpen = targetBox.classList.contains('open');
        targetBox.classList.toggle('open', !isOpen);
        btn.setAttribute('aria-expanded', String(!isOpen));
    });
});

document.querySelectorAll('.help-close').forEach((closeBtn) => {
    closeBtn.addEventListener('click', () => {
        const targetId = closeBtn.getAttribute('data-help-close');
        const targetBox = document.getElementById(targetId);
        const toggleBtn = document.querySelector(`.help-btn[data-help-target="${targetId}"]`);
        if (targetBox) targetBox.classList.remove('open');
        if (toggleBtn) toggleBtn.setAttribute('aria-expanded', 'false');
    });
});

// Detecta se está rodando direto pelo Flask (:5000) ou aberto via arquivo local
function getApiBaseUrl() {
    return window.location.protocol.startsWith('http') && window.location.port === '5000'
        ? ''
        : 'http://localhost:5000';
}

if (linkCsvRaw) linkCsvRaw.href = `${getApiBaseUrl()}/api/dataset/csv`;
if (linkTrain) linkTrain.href = `${getApiBaseUrl()}/train`;

function formatBRL(valor) {
    return Number(valor).toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
}

let datasetCarregado = false;
if (btnToggleDataset && datasetDrawer) {
    btnToggleDataset.addEventListener('click', async () => {
        const aberto = datasetDrawer.style.display === 'block';
        if (aberto) {
            datasetDrawer.style.display = 'none';
            btnToggleDataset.innerText = 'Consultar catálogo padronizado (amostra CSV)';
            return;
        }

        datasetDrawer.style.display = 'block';
        btnToggleDataset.innerText = 'Ocultar catálogo padronizado';

        if (datasetCarregado) return;

        try {
            const resp = await fetch(`${getApiBaseUrl()}/api/dataset?limit=15`);
            const info = await resp.json();
            if (resp.ok && info.amostra) {
                const mapaTarget = { 0: '0 (Estudos)', 1: '1 (Trabalho)', 2: '2 (Jogos/3D)' };
                datasetMeta.innerText = `${info.fonte} — ${info.total_registros} anúncios (exibindo 15 primeiros)`;
                datasetTbody.innerHTML = info.amostra.map((row) => {
                    const tituloHtml = row.permalink
                        ? `<a href="${row.permalink}" target="_blank" rel="noopener noreferrer" style="color:#93c5fd;text-decoration:underline;">${row.titulo} ↗</a>`
                        : row.titulo;
                    return `
                    <tr>
                        <td><strong>${row.marca}</strong></td>
                        <td>${tituloHtml}</td>
                        <td>${formatBRL(row.preco)}</td>
                        <td>${row.ram_gb} GB</td>
                        <td>${row.ssd_gb} GB</td>
                        <td>${row.score_cpu}</td>
                        <td>${row.tem_gpu_dedicada === 1 ? 'Sim (1)' : 'Não (0)'}</td>
                        <td><strong>${mapaTarget[row.target] || row.target}</strong></td>
                    </tr>
                `;
                }).join('');
                datasetCarregado = true;
            } else {
                datasetTbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">Não foi possível carregar a amostra.</td></tr>';
            }
        } catch (err) {
            datasetTbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">Inicie o servidor Back-end (http://localhost:5000) para visualizar os dados.</td></tr>';
        }
    });
}

// Preenche o formulário rápido com um dos 3 perfis base
function preencherFormulario({ preco, ram_gb, ssd_gb, score_cpu, tem_gpu_dedicada }) {
    document.getElementById('preco').value = preco;
    document.getElementById('ram_gb').value = String(ram_gb);
    document.getElementById('ssd_gb').value = String(ssd_gb);
    document.getElementById('score_cpu').value = String(score_cpu);
    document.getElementById('tem_gpu_dedicada').value = String(tem_gpu_dedicada);
    resultDiv.style.display = 'none';
}

if (btnPerfilBasico) {
    btnPerfilBasico.addEventListener('click', () => {
        preencherFormulario({ preco: 2500, ram_gb: 8, ssd_gb: 256, score_cpu: 2, tem_gpu_dedicada: 0 });
    });
}

if (btnPerfilTrabalho) {
    btnPerfilTrabalho.addEventListener('click', () => {
        preencherFormulario({ preco: 4200, ram_gb: 16, ssd_gb: 512, score_cpu: 3, tem_gpu_dedicada: 0 });
    });
}

if (btnPerfilGamer) {
    btnPerfilGamer.addEventListener('click', () => {
        preencherFormulario({ preco: 6800, ram_gb: 16, ssd_gb: 512, score_cpu: 4, tem_gpu_dedicada: 1 });
    });
}

// Renderiza os 3 blocos: (1) Classificação, (2) Tendências de Mercado e (3) Top-5 Notebooks
function renderizarResultado(data) {
    const classe = data.classe ?? 0;
    const confiancaTxt = data.confianca !== null && data.confianca !== undefined
        ? `${(data.confianca * 100).toFixed(1)}%`
        : 'N/A';

    const tend = data.tendencias || {};
    const cfg = tend.configuracao_dominante || {};
    const orc = tend.analise_orcamento || {};
    const marcasHtml = (tend.marcas_dominantes || [])
        .map((m) => `${m.marca} (${m.participacao_pct}%)`)
        .join(' • ') || 'N/A';

    const recs = data.recomendacoes || [];
    const recsHtml = recs.map((nb) => {
        // Exibe a foto apenas se a API tiver retornado o thumbnail real (nunca inventa imagem)
        const temFoto = Boolean(nb.thumbnail && nb.thumbnail.startsWith('http'));
        const fotoHtml = temFoto
            ? `<img class="nb-thumb" src="${nb.thumbnail}" alt="${nb.titulo}" loading="lazy">`
            : '';

        const temLink = Boolean(nb.permalink && nb.permalink.startsWith('http'));
        const tituloHtml = temLink
            ? `<a class="nb-title-link" href="${nb.permalink}" target="_blank" rel="noopener noreferrer">#${nb.ranking} — ${nb.titulo} ↗</a>`
            : `<span class="nb-title">#${nb.ranking} — ${nb.titulo}</span>`;

        const botaoMlHtml = temLink
            ? `<a class="nb-ml-btn" href="${nb.permalink}" target="_blank" rel="noopener noreferrer">Ver no Mercado Livre ↗</a>`
            : '';

        return `
        <div class="nb-card">
            ${fotoHtml}
            <div class="nb-content">
                <div class="nb-top">
                    ${tituloHtml}
                    <span class="nb-price">${formatBRL(nb.preco)}</span>
                </div>
                <div class="nb-badges">
                    <span class="badge">Marca: <strong>${nb.marca}</strong></span>
                    <span class="badge">${nb.ram_gb} GB RAM</span>
                    <span class="badge">${nb.ssd_gb} GB SSD</span>
                    <span class="badge">CPU Score: ${nb.score_cpu}/5</span>
                    <span class="badge ${nb.tem_gpu_dedicada === 1 ? 'badge-gpu' : ''}">
                        ${nb.tem_gpu_dedicada === 1 ? 'GPU Dedicada (RTX/GTX)' : 'GPU Integrada'}
                    </span>
                    ${botaoMlHtml}
                </div>
                <div class="match-bar-wrap">
                    <span>Compatibilidade Cosseno: <strong>${nb.compatibilidade_pct.toFixed(1)}%</strong></span>
                    <div class="match-track">
                        <div class="match-fill" style="width: ${Math.min(100, Math.max(10, nb.compatibilidade_pct))}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        `;
    }).join('');

    resultDiv.innerHTML = `
        <div class="profile-banner perfil-${classe}">
            <div>
                <div style="font-size: 0.78rem; text-transform: uppercase; opacity: 0.85;">Perfil Ideal Classificado (${data.algoritmo || 'ML'})</div>
                <div style="font-size: 1.18rem; font-weight: 700;">${data.predicao}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.78rem; opacity: 0.85;">Grau de Confiança</div>
                <div style="font-size: 1.15rem; font-weight: 700;">${confiancaTxt}</div>
            </div>
        </div>

        <div class="section-block">
            <div class="section-title">
                <span>📊 Tendências do Mercado para o Perfil</span>
                <span style="font-size: 0.76rem; color: var(--text-muted); font-weight: 400;">
                    Baseado em ${tend.total_anuncios_perfil || 0} anúncios da categoria
                </span>
            </div>
            <div class="trends-grid">
                <div class="trend-card">
                    <div class="trend-label">Preço Médio da Categoria</div>
                    <div class="trend-value">${formatBRL(tend.preco_medio_categoria || 0)}</div>
                    <div class="trend-sub">Faixa: ${formatBRL(tend.preco_min_categoria || 0)} a ${formatBRL(tend.preco_max_categoria || 0)}</div>
                </div>
                <div class="trend-card">
                    <div class="trend-label">Marcas Dominantes</div>
                    <div class="trend-value" style="font-size: 0.88rem;">${marcasHtml}</div>
                    <div class="trend-sub">Maior presença nos anúncios ativos</div>
                </div>
                <div class="trend-card">
                    <div class="trend-label">Configuração Padrão</div>
                    <div class="trend-value">${cfg.ram_gb_tipica || 8} GB RAM • ${cfg.ssd_gb_tipico || 256} GB SSD</div>
                    <div class="trend-sub">CPU médio ${cfg.score_cpu_medio || 3}/5 • ${cfg.gpu_dedicada_pct ?? 0}% com GPU dedicada</div>
                </div>
                <div class="trend-card">
                    <div class="trend-label">Termômetro de Orçamento</div>
                    <div class="trend-value">${orc.diferenca_pct_media > 0 ? '+' : ''}${orc.diferenca_pct_media ?? 0}% vs. Média</div>
                    <div class="trend-sub">${orc.status || ''}</div>
                </div>
            </div>
        </div>

        <div class="section-block" style="margin-bottom: 0;">
            <div class="section-title">
                <span>💻 Top-5 Notebooks Recomendados (KNN Similaridade de Cosseno)</span>
            </div>
            <div class="nb-list">
                ${recsHtml}
            </div>
        </div>
    `;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const features = [
        parseFloat(document.getElementById('preco').value),
        parseFloat(document.getElementById('ram_gb').value),
        parseFloat(document.getElementById('ssd_gb').value),
        parseFloat(document.getElementById('score_cpu').value),
        parseFloat(document.getElementById('tem_gpu_dedicada').value)
    ];

    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<div class="section-block" style="text-align:center;">Analisando perfil, calculando similaridade de cosseno e tendências...</div>';

    const API_URL = `${getApiBaseUrl()}/api/predict`;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });

        const data = await response.json();

        if (response.ok) {
            renderizarResultado(data);
        } else {
            resultDiv.innerHTML = `<div class="erro-banner">Erro: ${data.erro || 'Requisição inválida.'}</div>`;
        }
    } catch (err) {
        resultDiv.innerHTML = '<div class="erro-banner">Falha ao conectar com o servidor Back-end (http://localhost:5000).</div>';
    }
});
