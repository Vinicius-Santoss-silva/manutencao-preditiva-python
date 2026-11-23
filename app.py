from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core import (FS, WINDOW, CLASSES, FEATURE_NAMES, data_path,
                  demo_signal, features, load_record, spectrum, train_cross_load)
from download_data import download

st.set_page_config(page_title="VibraSense | IA e Engenharia Mecânica", page_icon="⚙️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.stApp{background:#07111e;color:#edf5ff}
.block-container{padding-top:2rem}
div[data-testid="stMetric"]{background:#102035;border:1px solid #234254;border-radius:14px;padding:14px}
div[data-testid="stSidebar"]{background:#0c1b2c}
h1,h2,h3{color:#f1f7ff!important}
.hero{padding:26px;border:1px solid #22516a;border-radius:18px;background:linear-gradient(110deg,#12344d,#101b32)}
.hero p{color:#bed2df}
</style>""", unsafe_allow_html=True)
DATA=Path(__file__).resolve().parent/"data"

@st.cache_data(show_spinner=False)
def signal(label, load, demo):
    if demo:
        return demo_signal(label, rpm={0:1797,1:1772,2:1750,3:1730}[load])
    return load_record(data_path(DATA,label,load),label,load)

@st.cache_resource(show_spinner="Treinando modelo a partir de registros experimentais...")
def train_model():
    return train_cross_load(DATA,0,1)

def figure(x,y,x_title,y_title,color="#36d5bc"):
    fig=go.Figure(go.Scatter(x=x,y=y,mode="lines",line=dict(color=color,width=1.8)))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#101e30",plot_bgcolor="#101e30",
                      margin=dict(l=28,r=16,t=20,b=38),height=320,showlegend=False,
                      xaxis_title=x_title,yaxis_title=y_title)
    return fig

st.markdown("""<div class="hero"><small>VIBRASENSE 2.0 · LABORATÓRIO DE CONDIÇÃO MECÂNICA</small>
<h1>Máquinas falam. Os dados ajudam a ouvir.</h1>
<p>Engenharia Mecânica + Ciência de Dados: análise de vibrações, FFT e classificação experimental de defeitos em rolamentos.</p></div>""",unsafe_allow_html=True)
st.caption("Projeto educacional. Não é monitoramento ao vivo nem ferramenta validada para manutenção industrial.")

available=[(label,load) for label in CLASSES for load in range(4) if data_path(DATA,label,load).exists()]
with st.sidebar:
    st.header("Configurações")
    st.caption("Nenhuma instalação local é necessária: este aplicativo roda na nuvem.")
    st.subheader("Dados experimentais CWRU")
    st.write(f"{len(available)} de 16 arquivos disponíveis neste ambiente.")
    if st.button("📥 Carregar medições reais (0 e 1 HP)",use_container_width=True):
        with st.spinner("Baixando arquivos da universidade; a transferência pode demorar..."):
            successes, errors=download(loads=(0,1),dest=DATA)
        st.cache_data.clear()
        st.cache_resource.clear()
        if errors:
            st.error("Alguns arquivos não puderam ser obtidos: "+", ".join(f"{x}: {e}" for x,e in errors))
        else:
            st.success(f"{len(successes)} arquivos verificados.")
        st.rerun()
    source=st.radio("Fonte do sinal",["Demonstração simulada","Medições CWRU"],
                    index=1 if available else 0)
    demo=source=="Demonstração simulada"
    if not demo and not available:
        st.warning("Use o botão de download acima para carregar os dados da universidade.")
        st.stop()
    labels=list(CLASSES) if demo else [c for c in CLASSES if any(c==a for a,_ in available)]
    label=st.selectbox("Condição do rolamento",labels)
    loads=list(range(4)) if demo else [i for i in range(4) if (label,i) in available]
    load=st.select_slider("Carga do motor (HP)",options=loads,value=loads[0])
    st.divider()
    st.info("Simulação didática: sinais sintéticos." if demo else
            "Medições reais de bancada experimental CWRU, não de máquinas em campo.")
    st.markdown("[Fonte oficial da CWRU](https://engineering.case.edu/bearingdatacenter/download-data-file)")

wave,rpm=signal(label,load,demo)
index=st.sidebar.slider("Janela temporal",0,max(0,len(wave)//WINDOW-1),0)
w=wave[index*WINDOW:(index+1)*WINDOW]
feat=features(w)
freq,db=spectrum(w)
tab1,tab2,tab3,tab4=st.tabs(["📊 Monitoramento","🔎 Comparar defeitos","🧠 Modelo de IA","📚 Metodologia"])
with tab1:
    st.subheader("Sinais e indicadores")
    cols=st.columns(4)
    cols[0].metric("Rotação",f"{rpm:.0f} RPM")
    cols[1].metric("RMS",f"{feat[0]:.3f}")
    cols[2].metric("Fator de crista",f"{feat[1]:.2f}")
    cols[3].metric("Curtose",f"{feat[3]:.2f}")
    st.caption("RMS na escala original do arquivo, sem unidade física calibrada; não é limiar de segurança.")
    c1,c2=st.columns(2)
    with c1:
        st.plotly_chart(figure(1000*np.arange(len(w))/FS,w,"Tempo (ms)","Amplitude (escala do arquivo)"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(figure(freq,db,"Frequência (Hz)","Densidade espectral (dB/Hz)"),
                        use_container_width=True)
    st.dataframe(pd.DataFrame({"Indicador":FEATURE_NAMES,"Valor":feat.round(4)}),
                 hide_index=True,use_container_width=True)
with tab2:
    st.subheader("Comparação de espectros")
    c1,c2=st.columns(2)
    with c1: first=st.selectbox("Condição A",labels,key="first")
    with c2: second=st.selectbox("Condição B",labels,index=min(1,len(labels)-1),key="second")
    def first_load(c):
        return load if demo or (c,load) in available else next(i for x,i in available if x==c)
    wa,_=signal(first,first_load(first),demo)
    wb,_=signal(second,first_load(second),demo)
    fa,pa=spectrum(wa[:WINDOW]);fb,pb=spectrum(wb[:WINDOW])
    fig=go.Figure()
    fig.add_scatter(x=fa,y=pa,mode="lines",name=first,line_color="#36d5bc")
    fig.add_scatter(x=fb,y=pb,mode="lines",name=second,line_color="#f8b763")
    fig.update_layout(template="plotly_dark",paper_bgcolor="#101e30",plot_bgcolor="#101e30",
                      xaxis_title="Frequência (Hz)",yaxis_title="Densidade espectral (dB/Hz)",
                      legend=dict(orientation="h"),height=380)
    st.plotly_chart(fig,use_container_width=True)
    if demo: st.warning("A comparação acima usa sinais sintetizados para demonstração.")
with tab3:
    st.subheader("Classificação de defeitos — Random Forest")
    st.markdown("**Treino:** registros de 0 HP. **Teste:** registros independentes de 1 HP.")
    required=[data_path(DATA,c,h) for c in CLASSES for h in (0,1)]
    missing=[p.name for p in required if not p.exists()]
    if missing:
        st.warning("Faltam arquivos reais: "+", ".join(missing))
        st.info("Clique em 'Carregar medições reais' na barra lateral. A simulação não é usada para treinar a IA.")
    else:
        if st.button("Treinar e avaliar modelo",type="primary"):
            st.session_state["trained"]=True
        if st.session_state.get("trained"):
            model,res=train_model()
            a,b,c=st.columns(3)
            a.metric("Acurácia",f"{100*res['acuracia']:.1f}%")
            b.metric("Acurácia balanceada",f"{100*res['acuracia_balanceada']:.1f}%")
            c.metric("Janelas de teste",res["amostras_teste"])
            fig=go.Figure(go.Heatmap(z=res["matriz"],x=list(CLASSES),y=list(CLASSES),
                                    text=res["matriz"],texttemplate="%{text}",colorscale="Teal",showscale=False))
            fig.update_layout(template="plotly_dark",height=380,xaxis_title="Predito",yaxis_title="Real")
            st.plotly_chart(fig,use_container_width=True)
            if not demo and load in (0,1):
                pred=int(model.predict(feat.reshape(1,-1))[0])
                st.info(f"Predição exploratória desta janela: **{CLASSES[pred]}**.")
            importance=pd.DataFrame({"Atributo":FEATURE_NAMES,"Importância":model.feature_importances_})
            st.bar_chart(importance.set_index("Atributo"))
            st.caption("Avaliação de laboratório: janelas do mesmo registro são correlacionadas; métricas não comprovam generalização industrial.")
with tab4:
    st.markdown("""
### O que o VibraSense faz
1. Lê sinais do acelerômetro do lado de acionamento da bancada CWRU, ou gera sinais didáticos claramente identificados.
2. Reamostra sinais normais originalmente a 48 kHz para 12 kHz com filtro antialiasing.
3. Calcula RMS, fator de crista, curtose, assimetria e características espectrais.
4. Calcula espectro de potência estimado por Welch (baseado em FFT).
5. Treina Random Forest em arquivos de carga 0 HP e testa em arquivos de carga 1 HP, sem misturar janelas dos mesmos arquivos entre treino e teste.

**Limites:** defeitos CWRU são experimentais; não há monitoramento ao vivo, valores calibrados de aceleração nem validação em máquinas de produção.

[Fonte acadêmica dos dados](https://engineering.case.edu/bearingdatacenter/download-data-file)
""")
st.caption("VibraSense 2.0 · Projeto de portfólio em Ciência de Dados e Engenharia Mecânica")
