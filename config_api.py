"""
Configurações e indicadores para consulta da API SPAECE
"""

import streamlit as st
from gerador_indicadores import gerar_todos_indicadores

# URL da API
API_URL = "https://avaliacaoemonitoramentoceara.caeddigital.net/portal/functions/getDadosResultado"

# Indicadores disponíveis - serão gerados dinamicamente baseado na rede
def get_indicadores(rede_selecionada="Estadual"):
    """Retorna os indicadores baseados na rede selecionada"""
    return gerar_todos_indicadores(rede_selecionada)

# Indicadores padrão (será sobrescrito quando necessário)
INDICADORES = get_indicadores("Estadual")

# Configurações removidas - sem filtros pré-definidos

# Parâmetros da aplicação
def get_app_config():
    """Obtém configurações da aplicação do arquivo secrets.toml"""
    try:
        # Tentar ler do secrets.toml
        secrets = st.secrets.get("api", {})
        session_token = secrets.get("session_token", "r:2aaf6c70b36f2f3ba29b35a5f021ab2c")
        installation_id = secrets.get("installation_id", "a0cfb234-b326-4ceb-aa2b-813625851c54")
        
        return {
            "_ApplicationId": "portal",
            "_ClientVersion": "js2.19.0",
            "_InstallationId": installation_id,
            "_SessionToken": session_token
        }
    except Exception as e:
        # Fallback para valores padrão em caso de erro
        return {
            "_ApplicationId": "portal",
            "_ClientVersion": "js2.19.0",
            "_InstallationId": "a0cfb234-b326-4ceb-aa2b-813625851c54",
            "_SessionToken": "r:2aaf6c70b36f2f3ba29b35a5f021ab2c"
        }

APP_CONFIG = get_app_config()

# Headers padrão para requisições
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
}


def criar_payload(
    indicadores=None,
    agregado="23",
    filtros=None,
    filtros_adicionais=None,
    nivel_abaixo="0"
):
    """
    Cria o payload para requisição à API
    
    Args:
        indicadores (list): Lista de códigos de indicadores
        agregado (str): Código do agregado
        filtros (list): Lista de filtros (etapa, disciplina, etc)
        filtros_adicionais (list): Filtros adicionais
        nivel_abaixo (str): Nível abaixo
    
    Returns:
        dict: Payload formatado para a API
    """
    if indicadores is None:
        indicadores = INDICADORES
    
    if filtros is None:
        filtros = []
    
    if filtros_adicionais is None:
        filtros_adicionais = []
    
    payload = {
        "CD_INDICADOR": indicadores,
        "agregado": agregado,
        "filtros": filtros,
        "filtrosAdicionais": filtros_adicionais,
        "nivelAbaixo": nivel_abaixo,
        "ordenacao": None,
        "collectionResultado": None,
        "CD_INDICADOR_LABEL": [],
        "TP_ENTIDADE_LABEL": "01",
        **APP_CONFIG
    }
    
    return payload