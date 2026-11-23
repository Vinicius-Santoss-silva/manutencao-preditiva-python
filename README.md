# ⚙️ VibraSense 2.0 — Engenharia Mecânica + Ciência de Dados

Aplicativo didático de análise de vibração de rolamentos com **Streamlit, NumPy, SciPy, Plotly e Scikit-learn**.

## 🔎 Recursos
- Visualização temporal da vibração e espectro de potência Welch / FFT
- RMS, fator de crista, curtose, assimetria e indicadores espectrais
- Comparação de condições: saudável, falha na pista interna, elemento rolante e pista externa
- Modo de demonstração com **sinais simulados claramente identificados**
- Download **sob demanda** das medições acadêmicas CWRU, sem arquivos pesados no repositório
- Random Forest com treino em registros de 0 HP e avaliação separada em 1 HP

## 🚀 Publicação gratuita no Streamlit Community Cloud

1. Abra https://share.streamlit.io/ com a conta GitHub autorizada.
2. Clique em **Create app** → **Deploy a public app from GitHub** (o texto pode variar).
3. Repository: `Vinicius-Santoss-silva/manutencao-preditiva-python`
4. Branch: `main`
5. Main file path: `app.py`
6. Clique em **Deploy**. O serviço instala automaticamente as bibliotecas do `requirements.txt`.

A página começa em **Demonstração simulada**. Para obter os registros experimentais, clique em **Carregar medições reais (0 e 1 HP)** na barra lateral. Os downloads são feitos no ambiente da aplicação; podem desaparecer quando o servidor reiniciar e precisam estar acessíveis pela rede. Se a fonte bloquear a transferência, o modo de demonstração continuará disponível.

## 💻 Execução local (opcional)
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## 📁 Organização
- `app.py`: interface interativa
- `core.py`: processamento de vibrações e classificador
- `download_data.py`: download condicionado das medições CWRU
- `requirements.txt`: dependências de execução
- `.streamlit/config.toml`: aparência da interface

## 🧪 Origem, limitações e ética técnica

Fonte: [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file), especialmente [normal baseline](https://engineering.case.edu/bearingdatacenter/normal-baseline-data) e [12 kHz drive end](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data).

São registros de **bancada experimental** com defeitos induzidos, não dados capturados em fábrica. Os registros de rolamento saudável foram capturados originalmente a 48 kHz e reamostrados para 12 kHz. A classificação usa janelas correlacionadas em cada arquivo, com separação por condição de carga; portanto, a métrica obtida é **exploratória e não representa desempenho industrial**. Amplitudes dos arquivos não estão calibradas no painel em unidades físicas.

**Não utilizar para decisões reais de manutenção ou segurança.**

Projeto autoral de demonstração para portfólio. Créditos pelas medições: CWRU.
