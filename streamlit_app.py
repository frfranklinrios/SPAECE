"""
Aplicação Streamlit para Consulta e Análise de Dados SPAECE

Esta aplicação permite consultar dados da API SPAECE (Sistema Permanente de Avaliação da Educação Básica do Ceará)
e realizar análises visuais dos dados de proficiência, participação, desempenho e habilidades dos estudantes.

Funcionalidades principais:
- Consulta de dados por código de agregado
- Análise de taxa de participação
- Visualização de proficiência média
- Distribuição por padrão de desempenho
- Análise de habilidades específicas
- Dados contextuais por etnia, NSE e sexo
- Exportação de dados em CSV e JSON

Autor: Sistema SPAECE
Data: 2024
"""

import streamlit as st
import requests
import pandas as pd
import json
import base64
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Paleta de cores personalizada
PALETA_CORES = [
    "#26a737",  # Verde médio vibrante
    "#f59c00",  # Laranja forte / dourado
    "#e94f0e",  # Laranja avermelhado intenso
    "#5db12f",  # Verde claro natural
    "#46ac33",  # Verde médio
    "#45b16e",  # Verde esmeralda suave
    "#e06a0c",  # Laranja queimado
    "#e4a500",  # Amarelo-ouro escuro
    "#2db39e",  # Verde água / turquesa
    "#fccf05"   # Amarelo vibrante
]

# Cores principais do sistema
COR_PRIMARIA = PALETA_CORES[0]  # Verde médio vibrante
COR_SECUNDARIA = PALETA_CORES[1]  # Laranja forte / dourado
COR_ACENTO = PALETA_CORES[2]  # Laranja avermelhado intenso
COR_SUCESSO = PALETA_CORES[3]  # Verde claro natural
COR_AVISO = PALETA_CORES[4]  # Verde médio
COR_INFO = PALETA_CORES[5]  # Verde esmeralda suave
COR_DANGER = PALETA_CORES[6]  # Laranja queimado
COR_WARNING = PALETA_CORES[7]  # Amarelo-ouro escuro
COR_LIGHT = PALETA_CORES[8]  # Verde água / turquesa
COR_BRIGHT = PALETA_CORES[9]  # Amarelo vibrante

import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config_api import API_URL, INDICADORES, HEADERS, criar_payload

# ==================== FUNÇÃO DE PROCESSAMENTO DE MARKDOWN COM RAG ====================

def extrair_texto_md(caminho_arquivo):
    """
    Extrai texto de um arquivo Markdown (.md)
    """
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
            texto_completo = arquivo.read()
        return texto_completo
    except Exception as e:
        st.error(f"Erro ao processar arquivo Markdown {caminho_arquivo}: {e}")
        return None

def processar_md_com_rag(texto_md):
    """
    Processa o arquivo Markdown usando técnicas de RAG para extrair informações relevantes
    """
    try:
        # Dividir o texto em chunks menores para melhor processamento
        chunks = dividir_em_chunks(texto_md, tamanho_chunk=1000, sobreposicao=200)
        
        # Extrair tabelas do final do arquivo
        tabelas = extrair_tabelas_do_md(texto_md)
        
        # Extrair seções importantes
        secoes_importantes = extrair_secoes_importantes(texto_md)
        
        # Criar índice de similaridade
        indice_similaridade = criar_indice_similaridade(chunks)
        
        return {
            'chunks': chunks,
            'tabelas': tabelas,
            'secoes_importantes': secoes_importantes,
            'indice_similaridade': indice_similaridade,
            'texto_completo': texto_md
        }
    except Exception as e:
        st.error(f"Erro ao processar arquivo Markdown: {e}")
        return None

def dividir_em_chunks(texto, tamanho_chunk=1000, sobreposicao=200):
    """
    Divide o texto em chunks menores para processamento RAG
    """
    palavras = texto.split()
    chunks = []
    
    for i in range(0, len(palavras), tamanho_chunk - sobreposicao):
        chunk = ' '.join(palavras[i:i + tamanho_chunk])
        if chunk.strip():
            chunks.append({
                'texto': chunk,
                'indice': len(chunks),
                'posicao_inicial': i
            })
    
    return chunks

def extrair_tabelas_do_md(texto_md):
    """
    Extrai tabelas do arquivo Markdown usando regex
    """
    try:
        # Procurar por padrões de tabelas no arquivo Markdown
        # Padrão para encontrar tabelas com dados numéricos
        padrao_tabela = r'(\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)*)'
        
        # Dividir o texto em seções para encontrar tabelas
        secoes = texto_md.split('\n---')
        ultimas_secoes = secoes[-5:] if len(secoes) > 5 else secoes
        
        tabelas_encontradas = []
        
        for secao in ultimas_secoes:
            # Procurar por padrões de tabela
            matches = re.findall(padrao_tabela, secao)
            if matches:
                # Criar conteúdo da tabela com os dados encontrados
                conteudo_tabela = f"Dados numéricos encontrados: {', '.join(matches[:10])}"
                tabelas_encontradas.append({
                    'conteudo': conteudo_tabela,
                    'secao': secao[:200] + '...' if len(secao) > 200 else secao,
                    'dados_numericos': matches[:10]  # Limitar a 10 matches por tabela
                })
        
        return tabelas_encontradas
    except Exception as e:
        st.error(f"Erro ao extrair tabelas do Markdown: {e}")
        return []

def extrair_secoes_importantes(texto_md):
    """
    Extrai seções importantes do arquivo Markdown como metodologia, indicadores, etc.
    """
    secoes = {}
    
    # Padrões para encontrar seções importantes
    padroes_secoes = {
        'metodologia': r'(metodologia|método|procedimento)',
        'indicadores': r'(indicador|métrica|medida)',
        'resultados': r'(resultado|conclusão|achado)',
        'recomendacoes': r'(recomenda|sugestão|orientação)',
        'tabelas': r'(tabela|quadro|dados)',
        'graficos': r'(gráfico|figura|chart)',
        'habilidades': r'(habilidade|competência|capacidade)',
        'componentes': r'(componente|disciplina|área)',
        'relacoes': r'(relação|relacionamento|conexão|vinculação)',
        'proficiencia': r'(proficiência|desempenho|rendimento)',
        'avaliacao': r'(avaliação|teste|exame)',
        'curriculo': r'(currículo|conteúdo|programa)',
        'bncc_competencias': r'(competência geral|competência específica|habilidade essencial)',
        'bncc_campos': r'(campo de experiência|área de conhecimento)',
        'bncc_objetivos': r'(objetivo de aprendizagem|expectativa de aprendizagem)',
        'bncc_etapas': r'(educação infantil|ensino fundamental|ensino médio)',
        'bncc_areas': r'(linguagens|matemática|ciências|humanas)',
        'bncc_objetivos_gerais': r'(objetivo geral|finalidade|propósito)',
        'bncc_principios': r'(princípio|fundamento|base)',
        'bncc_organizacao': r'(organização|estrutura|distribuição)',
        'bncc_avaliacao': r'(avaliação formativa|avaliação diagnóstica|avaliação somativa)',
        'dcrc_competencias_especificas': r'(competência específica|habilidade específica|descrição da habilidade)',
        'dcrc_descricoes_habilidades': r'(descrição|caracterização|definição.*habilidade)',
        'dcrc_relacoes_habilidades': r'(relação.*habilidade|vinculação.*competência|conexão.*componente)'
    }
    
    for nome_secao, padrao in padroes_secoes.items():
        matches = re.finditer(padrao, texto_md, re.IGNORECASE)
        for match in matches:
            # Extrair contexto ao redor da palavra-chave
            inicio = max(0, match.start() - 500)
            fim = min(len(texto_md), match.end() + 500)
            contexto = texto_md[inicio:fim]
            
            if nome_secao not in secoes:
                secoes[nome_secao] = []
            secoes[nome_secao].append(contexto)
    
    return secoes

def criar_indice_similaridade(chunks):
    """
    Cria um índice de similaridade usando TF-IDF para busca semântica
    """
    try:
        if not chunks:
            return None
        
        # Extrair textos dos chunks
        textos = [chunk['texto'] for chunk in chunks]
        
        # Criar vetorizador TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Manter palavras em português
            ngram_range=(1, 2)
        )
        
        # Vetorizar textos
        tfidf_matrix = vectorizer.fit_transform(textos)
        
        return {
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'chunks': chunks
        }
    except Exception as e:
        print(f"Erro ao criar índice de similaridade: {e}")
        return None


def comparar_habilidades_competencias(dados_rag, nome_habilidade=""):
    """
    Compara descrições de habilidades com competências específicas do DCRC
    """
    try:
        if not dados_rag or not dados_rag.get('secoes_importantes'):
            return ""
        
        secoes = dados_rag['secoes_importantes']
        comparacao = ""
        
        # Extrair competências específicas do DCRC
        competencias_especificas = secoes.get('dcrc_competencias_especificas', [])
        descricoes_habilidades = secoes.get('dcrc_descricoes_habilidades', [])
        relacoes_habilidades = secoes.get('dcrc_relacoes_habilidades', [])
        
        if competencias_especificas or descricoes_habilidades:
            comparacao = "\n\n===== ANÁLISE DE HABILIDADES COM BASE NAS RELAÇÕES E COMPETÊNCIAS BNCC/DCRC =====\n"
            
            # Adicionar competências específicas encontradas
            if competencias_especificas:
                comparacao += "\n🎯 COMPETÊNCIAS ESPECÍFICAS IDENTIFICADAS NOS DOCUMENTOS BNCC/DCRC:\n"
                for i, comp in enumerate(competencias_especificas[:3], 1):
                    # Identificar se é do BNCC ou DCRC
                    fonte = "BNCC" if "BNCC" in comp or "Base Nacional Comum Curricular" in comp else "DCRC"
                    comparacao += f"{i}. [{fonte}] {comp[:400]}...\n\n"
            
            # Adicionar descrições de habilidades
            if descricoes_habilidades:
                comparacao += "\n📝 DESCRIÇÕES DE HABILIDADES ENCONTRADAS NOS DOCUMENTOS BNCC/DCRC:\n"
                for i, desc in enumerate(descricoes_habilidades[:3], 1):
                    # Identificar se é do BNCC ou DCRC
                    fonte = "BNCC" if "BNCC" in desc or "Base Nacional Comum Curricular" in desc else "DCRC"
                    comparacao += f"{i}. [{fonte}] {desc[:400]}...\n\n"
            
            # Adicionar relações entre habilidades
            if relacoes_habilidades:
                comparacao += "\n🔗 RELAÇÕES ENTRE HABILIDADES IDENTIFICADAS NOS DOCUMENTOS BNCC/DCRC:\n"
                for i, rel in enumerate(relacoes_habilidades[:2], 1):
                    # Identificar se é do BNCC ou DCRC
                    fonte = "BNCC" if "BNCC" in rel or "Base Nacional Comum Curricular" in rel else "DCRC"
                    comparacao += f"{i}. [{fonte}] {rel[:400]}...\n\n"
            
            # Instruções específicas para análise de habilidades com foco em relações
            comparacao += """
🔧 INSTRUÇÕES OBRIGATÓRIAS PARA ANÁLISE DE HABILIDADES:

1. PROXIMIDADE ENTRE HABILIDADES:
   - IDENTIFIQUE habilidades que aparecem próximas nos dados
   - ANALISE se habilidades com desempenho similar estão relacionadas
   - EXPLIQUE por que certas habilidades têm padrões similares
   - SUGIRA intervenções que trabalhem habilidades relacionadas juntas

2. RELAÇÃO DENTRO DO PRÓPRIO COMPONENTE:
   - FOQUE nas habilidades que pertencem ao mesmo componente
   - IDENTIFIQUE hierarquias dentro do componente
   - ANALISE dependências entre habilidades do mesmo componente
   - SUGIRA sequências de ensino baseadas nas relações internas

3. RELAÇÃO ENTRE COMPONENTES:
   - MAPEIE como habilidades de diferentes componentes se conectam
   - IDENTIFIQUE competências que dependem de múltiplos componentes
   - ANALISE transferências de conhecimento entre componentes
   - SUGIRA abordagens interdisciplinares baseadas nas relações

4. COMPETÊNCIAS ESPECÍFICAS:
   - RELACIONE cada habilidade com competências específicas do DCRC
   - IDENTIFIQUE quais competências são mais críticas
   - ANALISE lacunas entre habilidades e competências esperadas
   - SUGIRA desenvolvimento de competências específicas

5. DESCRIÇÕES DAS HABILIDADES:
   - USE as descrições do DCRC para entender o que cada habilidade envolve
   - COMPARE descrições com desempenho real nos dados
   - IDENTIFIQUE habilidades mal compreendidas pelos estudantes
   - SUGIRA reformulações pedagógicas baseadas nas descrições

6. ANÁLISE INTEGRADA:
   - COMBINE proximidade, relações e competências na análise
   - IDENTIFIQUE padrões complexos de desempenho
   - SUGIRA intervenções sistêmicas baseadas nas relações
   - MONITORE progresso considerando as interconexões
"""
        
        return comparacao
    except Exception as e:
        print(f"Erro na comparação habilidades-competências: {e}")
        return ""

def analisar_percursos_aprendizado(dados_rag, nome_habilidade=""):
    """
    Analisa percursos de aprendizado, dependências e relações entre habilidades de forma CIRÚRGICA
    """
    try:
        if not dados_rag or not dados_rag.get('indice_similaridade'):
            return ""
        
        # Buscar informações específicas sobre percursos de aprendizado
        consultas_percurso = [
            f"percurso aprendizado progressão sequência {nome_habilidade}",
            f"dependência pré-requisito hierarquia habilidade {nome_habilidade}",
            f"relação conexão vinculação habilidade componente {nome_habilidade}",
            f"competência específica objetivo aprendizagem {nome_habilidade}",
            f"metodologia estratégia ensino habilidade {nome_habilidade}"
        ]
        
        contexto_percursos = "\n\n===== ANÁLISE HIERÁRQUICA DE PERCURSOS DE APRENDIZADO =====\n"
        
        for consulta in consultas_percurso:
            informacoes = buscar_informacoes_relevantes(consulta, dados_rag, top_k=3)
            if informacoes:
                contexto_percursos += f"\n🔍 INFORMAÇÕES SOBRE: {consulta.upper()}\n"
                for info in informacoes:
                    fonte = info.get('fonte', 'Documento')
                    contexto_percursos += f"[{fonte}] {info['texto'][:300]}...\n\n"
        
        # Instruções HIERÁRQUICAS para análise de percursos
        contexto_percursos += """
🎯 INSTRUÇÕES HIERÁRQUICAS PARA ANÁLISE DE PERCURSOS DE APRENDIZADO:

**ANÁLISE HIERÁRQUICA OBRIGATÓRIA - PERSPECTIVA POR NÍVEL EDUCACIONAL:**

1. MAPEAMENTO HIERÁRQUICO DE DEPENDÊNCIAS:
   - IDENTIFIQUE EXATAMENTE quais habilidades são pré-requisito para outras conforme BNCC/DCRC
   - MAPEIE a hierarquia ESPECÍFICA: habilidades básicas → intermediárias → avançadas
   - ANALISE habilidades "gargalo" ESPECÍFICAS que bloqueiam o desenvolvimento de outras
   - IDENTIFIQUE habilidades que se reforçam mutuamente de forma CONCRETA
   - SUGIRA sequências de ensino ESPECÍFICAS baseadas nas dependências identificadas
   - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores

2. PERCURSOS HIERÁRQUICOS ESTRUTURADOS:
   - DESENHE percursos de aprendizado ESPECÍFICOS: quais habilidades devem ser desenvolvidas primeiro
   - MAPEIE pontos de convergência CONCRETOS onde múltiplas habilidades se encontram
   - IDENTIFIQUE competências ESPECÍFICAS que dependem de múltiplos componentes
   - ANALISE transferências de conhecimento ESPECÍFICAS entre componentes
   - SUGIRA abordagens interdisciplinares ESPECÍFICAS baseadas nos percursos
   - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores

3. RELAÇÕES HIERÁRQUICAS ENTRE HABILIDADES:
   - IDENTIFIQUE habilidades que aparecem próximas nos dados ESPECÍFICOS
   - ANALISE se habilidades com desempenho similar estão relacionadas de forma CONCRETA
   - EXPLIQUE por que certas habilidades têm padrões similares de forma ESPECÍFICA
   - MAPEIE como habilidades de diferentes componentes se conectam de forma CONCRETA
   - SUGIRA intervenções ESPECÍFICAS que trabalhem habilidades relacionadas juntas
   - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores

4. COMPETÊNCIAS E OBJETIVOS HIERÁRQUICOS:
   - RELACIONE cada habilidade com competências específicas do BNCC/DCRC de forma CONCRETA
   - IDENTIFIQUE objetivos de aprendizagem ESPECÍFICOS para cada habilidade
   - ANALISE lacunas ESPECÍFICAS entre habilidades e competências esperadas
   - MAPEIE competências gerais da BNCC desenvolvidas através das habilidades de forma CONCRETA
   - SUGIRA desenvolvimento de competências ESPECÍFICO baseado nos documentos
   - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores

5. METODOLOGIAS E ESTRATÉGIAS HIERÁRQUICAS:
   - USE metodologias ESPECÍFICAS sugeridas nos documentos BNCC/DCRC para cada habilidade
   - IDENTIFIQUE recursos e materiais ESPECÍFICOS recomendados nos documentos
   - MAPEIE estratégias ESPECÍFICAS para desenvolvimento de cada habilidade
   - SUGIRA reformulações pedagógicas ESPECÍFICAS baseadas nas descrições dos documentos
   - IDENTIFIQUE práticas de linguagem e campos de experiência ESPECÍFICOS relevantes
   - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores

6. INTERVENÇÕES HIERÁRQUICAS SISTÊMICAS:
   - DESENHE planos de ação ESPECÍFICOS baseados nos percursos de aprendizado identificados
   - IDENTIFIQUE pontos de intervenção mais eficazes de forma CONCRETA baseado nas dependências
   - MAPEIE como melhorar uma habilidade impacta outras habilidades de forma ESPECÍFICA
   - SUGIRA intervenções sistêmicas ESPECÍFICAS baseadas nas relações identificadas
   - MONITORE progresso considerando as interconexões de forma CONCRETA conforme BNCC/DCRC
   - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores

**REFERENCIAMENTO HIERÁRQUICO OBRIGATÓRIO:**
- REFERENCIE SEMPRE: "Conforme a BNCC", "Segundo o DCRC", "Baseado nos percursos identificados"
- CITE competências específicas e objetivos de aprendizagem mencionados nos documentos de forma CONCRETA
- REFERENCIE metodologias e recursos sugeridos nos documentos de forma ESPECÍFICA
- IDENTIFIQUE campos de experiência e práticas de linguagem dos documentos de forma CONCRETA
- DIFERENCIE entre informações dos documentos vs. análises genéricas de forma CLARA
- SEJA ESPECÍFICO: evite generalizações, foque nos dados específicos da entidade
- **CITE OBRIGATORIAMENTE BNCC E DCRC**: Sempre que possível, referencie tanto a BNCC quanto o DCRC como fontes principais das metodologias, competências e diretrizes curriculares
- **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores
"""
        
        return contexto_percursos
        
    except Exception as e:
        print(f"Erro na análise de percursos de aprendizado: {e}")
        return ""

def gerar_analise_personalizada(dados_rag, df_info, nome_grafico, contexto_especifico=""):
    """
    Gera análise personalizada baseada nos dados específicos da entidade e gráfico
    """
    try:
        if not dados_rag or not dados_rag.get('secoes_importantes'):
            return ""
        
        secoes = dados_rag['secoes_importantes']
        analise_personalizada = ""
        
        # Extrair dados específicos do DataFrame
        estatisticas = df_info.get('estatisticas', {})
        amostra_dados = df_info.get('amostra_dados', [])
        debug_info = df_info.get('debug_info', {})
        
        # Identificar padrões específicos nos dados
        padroes_identificados = []
        if estatisticas:
            for coluna, stats in estatisticas.items():
                if isinstance(stats, dict):
                    if 'mean' in stats and stats['mean'] < 50:
                        padroes_identificados.append(f"Baixo desempenho em {coluna} (média: {stats['mean']:.1f})")
                    elif 'mean' in stats and stats['mean'] > 80:
                        padroes_identificados.append(f"Alto desempenho em {coluna} (média: {stats['mean']:.1f})")
        
        # Extrair recomendações específicas dos documentos DCRC + BNCC
        recomendacoes = secoes.get('recomendacoes', [])
        metodologia = secoes.get('metodologia', [])
        bncc_competencias = secoes.get('bncc_competencias', [])
        bncc_objetivos = secoes.get('bncc_objetivos', [])
        dcrc_competencias_especificas = secoes.get('dcrc_competencias_especificas', [])
        dcrc_descricoes_habilidades = secoes.get('dcrc_descricoes_habilidades', [])
        
        if padroes_identificados or recomendacoes or metodologia:
            analise_personalizada = "\n\n===== ANÁLISE PERSONALIZADA COM BASE NOS DOCUMENTOS DCRC + BNCC =====\n"
            
            # Padrões identificados nos dados
            if padroes_identificados:
                analise_personalizada += "\n🔍 PADRÕES IDENTIFICADOS NOS DADOS:\n"
                for padrao in padroes_identificados[:5]:
                    analise_personalizada += f"• {padrao}\n"
            
            # Informações específicas dos documentos encontradas
            if recomendacoes or metodologia or bncc_competencias or dcrc_competencias_especificas:
                analise_personalizada += "\n📚 INFORMAÇÕES ESPECÍFICAS DOS DOCUMENTOS ENCONTRADAS:\n"
                
                if recomendacoes:
                    analise_personalizada += f"• DCRC - Recomendações: {len(recomendacoes)} seções encontradas\n"
                if metodologia:
                    analise_personalizada += f"• DCRC - Metodologia: {len(metodologia)} seções encontradas\n"
                if bncc_competencias:
                    analise_personalizada += f"• BNCC - Competências: {len(bncc_competencias)} seções encontradas\n"
                if dcrc_competencias_especificas:
                    analise_personalizada += f"• DCRC - Competências Específicas: {len(dcrc_competencias_especificas)} seções encontradas\n"
                if dcrc_descricoes_habilidades:
                    analise_personalizada += f"• DCRC - Descrições de Habilidades: {len(dcrc_descricoes_habilidades)} seções encontradas\n"
            
            # Recomendações específicas baseadas nos padrões E documentos
            if padroes_identificados:
                analise_personalizada += "\n💡 RECOMENDAÇÕES ESPECÍFICAS (BASEADAS NOS DOCUMENTOS):\n"
                for padrao in padroes_identificados[:3]:
                    if "Baixo desempenho" in padrao:
                        analise_personalizada += f"• Para {padrao}: Implementar intervenção pedagógica específica baseada nas competências específicas do DCRC identificadas\n"
                    elif "Alto desempenho" in padrao:
                        analise_personalizada += f"• Para {padrao}: Manter e expandir práticas exitosas, compartilhar com outras áreas usando metodologias do DCRC\n"
            
            # Ações específicas baseadas no tipo de gráfico, dados E documentos
            if 'habilidade' in nome_grafico.lower():
                analise_personalizada += """
🎯 AÇÕES ESPECÍFICAS PARA HABILIDADES (BASEADAS NOS DOCUMENTOS):
• Analisar quais habilidades específicas têm baixo desempenho nos dados
• Criar planos de intervenção direcionados usando as competências específicas do DCRC
• Desenvolver atividades práticas baseadas nas descrições de habilidades do DCRC
• Estabelecer grupos de estudo focados nas habilidades com menor desempenho
• Monitorar progresso usando indicadores específicos do DCRC
• Alinhar com competências gerais e específicas da BNCC identificadas
"""
            elif 'proficiência' in nome_grafico.lower():
                analise_personalizada += """
📊 AÇÕES ESPECÍFICAS PARA PROFICIÊNCIA (BASEADAS NOS DOCUMENTOS):
• Identificar níveis de proficiência específicos nos dados
• Criar planos de intervenção usando metodologias do DCRC
• Estabelecer metas de proficiência baseadas nos objetivos da BNCC
• Implementar avaliação formativa contínua com foco nas competências específicas
• Desenvolver estratégias de recuperação baseadas nas recomendações do DCRC
• Alinhar com campos de experiência da BNCC identificados
"""
            elif 'participação' in nome_grafico.lower():
                analise_personalizada += """
👥 AÇÕES ESPECÍFICAS PARA PARTICIPAÇÃO (BASEADAS NOS PDFs):
• Analisar taxa de participação específica nos dados
• Identificar fatores que impactam a participação usando metodologias do DCRC
• Criar estratégias de engajamento baseadas nos princípios da BNCC
• Estabelecer parcerias com famílias usando orientações do DCRC
• Monitorar participação com indicadores específicos do DCRC
• Alinhar com objetivos de aprendizagem da BNCC
"""
            
            # Instruções específicas para análise personalizada COM PDFs
            analise_personalizada += """
🔧 INSTRUÇÕES PARA ANÁLISE PERSONALIZADA COM PDFs:
1. FOQUE nos dados específicos da entidade analisada
2. IDENTIFIQUE padrões únicos nos dados apresentados
3. RELACIONE os dados com as competências específicas do DCRC encontradas
4. SUGIRA ações baseadas nos dados reais E nas informações dos PDFs
5. CONSIDERE o contexto específico da entidade
6. MONITORE indicadores específicos identificados nos dados
7. ADAPTE as ações conforme os dados específicos E os PDFs
8. AVALIE o progresso com base nos dados apresentados E nas metodologias do DCRC
9. REFERENCIE explicitamente as informações dos PDFs nas análises
10. DIFERENCIE claramente quando está usando informações dos PDFs vs. análises genéricas
"""
        
        return analise_personalizada
    except Exception as e:
        print(f"Erro na geração de análise personalizada: {e}")
        return ""

def gerar_acoes_escola_baseadas_pdfs(dados_rag, tipo_grafico, contexto_especifico=""):
    """
    Gera ações específicas que a escola deve tomar baseadas nos PDFs, com foco na educação básica
    """
    try:
        if not dados_rag or not dados_rag.get('secoes_importantes'):
            return ""
        
        secoes = dados_rag['secoes_importantes']
        acoes_escola = ""
        
        # Extrair recomendações e orientações dos PDFs
        recomendacoes = secoes.get('recomendacoes', [])
        metodologia = secoes.get('metodologia', [])
        bncc_competencias = secoes.get('bncc_competencias', [])
        bncc_objetivos = secoes.get('bncc_objetivos', [])
        bncc_principios = secoes.get('bncc_principios', [])
        
        if recomendacoes or metodologia or bncc_competencias:
            acoes_escola = "\n\n===== AÇÕES ESPECÍFICAS PARA A ESCOLA (BASEADAS NOS PDFs) =====\n"
            
            # Ações baseadas no DCRC
            if recomendacoes or metodologia:
                acoes_escola += "\n📋 AÇÕES BASEADAS NO DCRC:\n"
                if recomendacoes:
                    for i, rec in enumerate(recomendacoes[:2], 1):
                        acoes_escola += f"• {rec[:300]}...\n"
                if metodologia:
                    for i, met in enumerate(metodologia[:2], 1):
                        acoes_escola += f"• {met[:300]}...\n"
            
            # Ações baseadas na BNCC
            if bncc_competencias or bncc_objetivos:
                acoes_escola += "\n📚 AÇÕES BASEADAS NA BNCC:\n"
                if bncc_competencias:
                    for i, comp in enumerate(bncc_competencias[:2], 1):
                        acoes_escola += f"• {comp[:300]}...\n"
                if bncc_objetivos:
                    for i, obj in enumerate(bncc_objetivos[:2], 1):
                        acoes_escola += f"• {obj[:300]}...\n"
            
            # Ações específicas por tipo de gráfico
            if 'habilidade' in tipo_grafico.lower():
                acoes_escola += """
🎯 AÇÕES ESPECÍFICAS PARA HABILIDADES:
• Implementar atividades práticas baseadas nas competências específicas do DCRC
• Criar sequências didáticas que desenvolvam habilidades inter-relacionadas
• Estabelecer momentos de reflexão sobre o desenvolvimento das competências
• Organizar grupos de estudo para habilidades com baixo desempenho
• Desenvolver materiais didáticos alinhados com as competências da BNCC
"""
            elif 'proficiência' in tipo_grafico.lower():
                acoes_escola += """
📊 AÇÕES ESPECÍFICAS PARA PROFICIÊNCIA:
• Alinhar práticas pedagógicas com os objetivos de aprendizagem da BNCC
• Implementar avaliação formativa contínua baseada nas competências
• Criar planos de intervenção para níveis de proficiência críticos
• Estabelecer metas de proficiência por competência específica
• Desenvolver estratégias de recuperação baseadas nas competências
"""
            elif 'participação' in tipo_grafico.lower():
                acoes_escola += """
👥 AÇÕES ESPECÍFICAS PARA PARTICIPAÇÃO:
• Implementar estratégias de engajamento baseadas nos princípios da BNCC
• Criar ambientes de aprendizagem que promovam participação ativa
• Desenvolver atividades que conectem com os campos de experiência
• Estabelecer parcerias com famílias baseadas nas orientações do DCRC
• Organizar momentos de protagonismo estudantil
"""
            
            # Instruções específicas para ações práticas
            acoes_escola += """
🔧 INSTRUÇÕES PARA IMPLEMENTAÇÃO:
1. PRIORIZE: Ações que desenvolvam competências básicas fundamentais
2. SEQUENCIE: Implemente ações em ordem de complexidade crescente
3. MONITORE: Acompanhe o progresso baseado nas competências específicas
4. ADAPTE: Ajuste as ações conforme o contexto da escola
5. COLABORE: Envolva toda a comunidade escolar nas ações
6. DOCUMENTE: Registre as ações e seus resultados
7. AVALIE: Use os indicadores do DCRC para avaliar o progresso
8. REFLITA: Promova reflexão coletiva sobre as práticas implementadas
"""
        
        return acoes_escola
    except Exception as e:
        print(f"Erro na geração de ações para escola: {e}")
        return ""

def buscar_informacoes_relevantes(consulta, dados_rag, top_k=5):
    """
    Busca informações relevantes no PDF usando RAG
    """
    try:
        if not dados_rag or not dados_rag.get('indice_similaridade'):
            return []
        
        indice = dados_rag['indice_similaridade']
        vectorizer = indice['vectorizer']
        tfidf_matrix = indice['tfidf_matrix']
        chunks = indice['chunks']
        
        # Expandir consulta com termos relacionados específicos
        if 'habilidade' in consulta.lower() or 'competência' in consulta.lower():
            consulta_expandida = f"{consulta} habilidade competência capacidade componente relação entre componentes proximidade habilidades SPAECE DCRC BNCC avaliação proficiência competência geral competência específica habilidade essencial descrição da habilidade caracterização habilidade específica vinculação competência conexão componente relação dentro próprio componente competências específicas descrições habilidades relações habilidades objeto de conhecimento campo de experiência prática de linguagem percurso aprendizado progressão sequência dependência pré-requisito hierarquia metodologia estratégia ensino objetivo aprendizagem expectativa aprendizagem direito aprendizagem base nacional comum curricular documento curricular referencial"
        elif 'proficiência' in consulta.lower() or 'desempenho' in consulta.lower():
            consulta_expandida = f"{consulta} proficiência desempenho rendimento SPAECE DCRC BNCC avaliação competência objetivo de aprendizagem competência específica"
        elif 'participação' in consulta.lower():
            consulta_expandida = f"{consulta} participação frequência presença SPAECE DCRC BNCC educação básica"
        else:
            consulta_expandida = f"{consulta} educação avaliação SPAECE DCRC BNCC metodologia indicadores competência geral competência específica habilidade essencial descrição habilidade"
        
        # Vetorizar a consulta
        consulta_vector = vectorizer.transform([consulta_expandida])
        
        # Calcular similaridade
        similaridades = cosine_similarity(consulta_vector, tfidf_matrix).flatten()
        
        # Obter top-k resultados mais similares
        top_indices = np.argsort(similaridades)[::-1][:top_k]
        
        resultados = []
        for idx in top_indices:
            if similaridades[idx] > 0.05:  # Threshold mais baixo para capturar mais informações
                chunk_texto = chunks[idx]['texto']
                # Identificar se o chunk é do BNCC ou DCRC
                if "BNCC" in chunk_texto or "Base Nacional Comum Curricular" in chunk_texto or "BNCC_20dez_site" in chunk_texto:
                    fonte_documento = "BNCC"
                elif "DCRC" in chunk_texto or "Documento Curricular Referencial" in chunk_texto or "dcrc" in chunk_texto.lower():
                    fonte_documento = "DCRC"
                else:
                    # Se não conseguir identificar, usar contexto do texto combinado
                    # Alternar entre BNCC e DCRC para dar equilíbrio
                    fonte_documento = "BNCC" if idx % 2 == 0 else "DCRC"
                
                resultados.append({
                    'chunk': chunks[idx],
                    'similaridade': similaridades[idx],
                    'texto': chunk_texto,
                    'fonte': fonte_documento
                })
        
        # Busca específica para habilidades e relações com foco em BNCC e DCRC
        if 'habilidade' in consulta.lower() or 'competência' in consulta.lower():
            palavras_habilidade = ['habilidade', 'competência', 'capacidade', 'componente', 'relação', 'vinculação', 'conexão', 'descrição', 'caracterização', 'específica', 'geral', 'essencial', 'dcrc', 'documento curricular', 'bncc', 'base nacional comum curricular']
            for i, chunk in enumerate(chunks):
                texto_chunk = chunk['texto'].lower()
                if any(palavra in texto_chunk for palavra in palavras_habilidade):
                    # Verificar se já não está nos resultados
                    if not any(r['chunk']['indice'] == chunk['indice'] for r in resultados):
                        # Identificar fonte para habilidades
                        if "BNCC" in chunk['texto'] or "Base Nacional Comum Curricular" in chunk['texto']:
                            fonte_habilidade = "BNCC"
                        elif "DCRC" in chunk['texto'] or "Documento Curricular Referencial" in chunk['texto']:
                            fonte_habilidade = "DCRC"
                        else:
                            # Alternar entre BNCC e DCRC para dar equilíbrio
                            fonte_habilidade = "BNCC" if i % 2 == 0 else "DCRC"
                        
                        resultados.append({
                            'chunk': chunk,
                            'similaridade': 0.4,  # Similaridade alta para habilidades
                            'texto': chunk['texto'],
                            'fonte': fonte_habilidade
                        })
                        if len(resultados) >= top_k * 2:  # Mais resultados para habilidades
                            break
        
        # Se não encontrou resultados específicos, buscar por palavras-chave gerais
        if not resultados:
            palavras_chave = consulta.lower().split()
            for i, chunk in enumerate(chunks):
                texto_chunk = chunk['texto'].lower()
                if any(palavra in texto_chunk for palavra in palavras_chave):
                    resultados.append({
                        'chunk': chunk,
                        'similaridade': 0.3,  # Similaridade artificial para palavras-chave
                        'texto': chunk['texto']
                    })
                    if len(resultados) >= top_k:
                        break
        
        return resultados
    except Exception as e:
        print(f"Erro na busca RAG: {e}")
        return []

def analisar_pdf_com_rag_groq(dados_rag, contexto_analise="", consulta_especifica=""):
    """
    Analisa o PDF usando RAG + Groq para encontrar informações específicas
    """
    try:
        # Configurar a API do Groq
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            return "❌ Chave da API Groq não configurada"
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Buscar informações relevantes usando RAG
        if consulta_especifica:
            informacoes_relevantes = buscar_informacoes_relevantes(consulta_especifica, dados_rag, top_k=3)
            contexto_rag = "\n\n".join([info['texto'] for info in informacoes_relevantes])
        else:
            # Usar seções importantes como contexto
            contexto_rag = ""
            for secao, conteudos in dados_rag.get('secoes_importantes', {}).items():
                contexto_rag += f"\n=== {secao.upper()} ===\n"
                contexto_rag += "\n".join(conteudos[:2])  # Primeiros 2 conteúdos de cada seção
        
        # Extrair tabelas para análise
        tabelas_contexto = ""
        if dados_rag.get('tabelas'):
            tabelas_contexto = "\n=== TABELAS E DADOS NUMÉRICOS ===\n"
            for i, tabela in enumerate(dados_rag['tabelas'][:3]):  # Primeiras 3 tabelas
                if isinstance(tabela, dict) and 'conteudo' in tabela:
                    tabelas_contexto += f"\nTabela {i+1}:\n{tabela['conteudo']}\n"
                else:
                    # Fallback para outras estruturas de tabela
                    conteudo_fallback = str(tabela) if tabela else "Sem conteúdo"
                    tabelas_contexto += f"\nTabela {i+1}:\n{conteudo_fallback}\n"
        
        # Preparar prompt otimizado com RAG
        prompt = f"""
        Analise os documentos BNCC e DCRC usando as informações mais relevantes encontradas:

        CONTEXTO DA ANÁLISE: {contexto_analise}

        INFORMAÇÕES RELEVANTES ENCONTRADAS NOS DOCUMENTOS BNCC E DCRC:
        {contexto_rag[:4000]}

        {tabelas_contexto[:2000]}

        INSTRUÇÕES CRÍTICAS - USE OBRIGATORIAMENTE OS DOCUMENTOS BNCC E DCRC:
        1. **FUNDAMENTE SUAS RESPOSTAS** exclusivamente nas informações dos documentos BNCC e DCRC apresentados acima
        2. **REFERENCIE EXPLICITAMENTE** quando usar informações do BNCC ("conforme a BNCC") ou DCRC ("segundo o DCRC")
        3. **CITE COMPETÊNCIAS ESPECÍFICAS** mencionadas nos documentos quando relevante
        4. **USE OBJETIVOS DE APRENDIZAGEM** e expectativas de aprendizagem dos documentos
        5. **RELACIONE COM CAMPOS DE EXPERIÊNCIA** e áreas de conhecimento da BNCC
        6. **APLIQUE METODOLOGIAS** sugeridas no DCRC para intervenções pedagógicas
        7. **CONSIDERE PRINCÍPIOS** e fundamentos da BNCC em suas recomendações
        8. **IDENTIFIQUE LACUNAS** entre desempenho atual e expectativas dos documentos
        9. **SUGIRA AÇÕES** baseadas nas diretrizes curriculares apresentadas
        10. **EVITE ANÁLISES GENÉRICAS** - seja específico com base nos documentos

        ESTRUTURA OBRIGATÓRIA DA RESPOSTA:
        1. **Fundamentação Documental**: Cite especificamente trechos dos documentos BNCC/DCRC
        2. **Análise Curricular**: Relacione os dados com competências e habilidades dos documentos
        3. **Recomendações Baseadas em Evidências**: Use metodologias dos documentos
        4. **Ações Pedagógicas Específicas**: Baseadas nas diretrizes curriculares
        5. **Indicadores de Progressão**: Alinhados com expectativas de aprendizagem

        IMPORTANTE: Sua análise deve ser fundamentada EXCLUSIVAMENTE nos documentos BNCC e DCRC apresentados. Evite análises genéricas ou baseadas em conhecimento geral.

        Responda em português brasileiro de forma clara, objetiva e acionável.
        """
        
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um especialista em análise de dados educacionais e avaliação da educação básica. Sua função é analisar dados SPAECE fundamentando-se EXCLUSIVAMENTE nos documentos BNCC (Base Nacional Comum Curricular) e DCRC (Documento Curricular Referencial do Ceará) fornecidos. Você deve citar explicitamente trechos dos documentos, referenciar competências específicas, habilidades e metodologias mencionadas nos documentos. Evite análises genéricas - seja específico e fundamentado nos documentos curriculares apresentados."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 3000,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ Erro na API Groq: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"❌ Erro na análise RAG do PDF: {str(e)}"

def analisar_pdf_com_groq(texto_pdf, contexto_analise=""):
    """
    Analisa o conteúdo de um PDF usando Groq (versão simples)
    """
    try:
        # Configurar a API do Groq
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            return "❌ Chave da API Groq não configurada"
        
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Preparar prompt para análise do PDF
        prompt = f"""
        Analise o seguinte documento PDF e forneça insights relevantes para análise educacional:

        CONTEXTO DA ANÁLISE: {contexto_analise}

        CONTEÚDO DO PDF:
        {texto_pdf[:8000]}  # Limitar tamanho para evitar token limit

        Por favor, forneça:
        1. Resumo dos principais pontos do documento
        2. Métricas e indicadores mencionados
        3. Recomendações ou insights educacionais
        4. Padrões ou tendências identificadas
        5. Sugestões para análise de dados SPAECE baseadas no documento

        Responda em português brasileiro de forma clara e objetiva.
        """
        
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um especialista em análise de dados educacionais e avaliação da educação básica. Sua função é analisar dados SPAECE fundamentando-se EXCLUSIVAMENTE nos documentos BNCC (Base Nacional Comum Curricular) e DCRC (Documento Curricular Referencial do Ceará) fornecidos. Você deve citar explicitamente trechos dos documentos, referenciar competências específicas, habilidades e metodologias mencionadas nos documentos. Evite análises genéricas - seja específico e fundamentado nos documentos curriculares apresentados."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"❌ Erro na API Groq: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"❌ Erro na análise do PDF: {str(e)}"

# ==================== FUNÇÃO PARA OBTER CONTEXTO DOS BANNERS ====================

def obter_contexto_seduc_spaece():
    """
    Retorna contexto específico da SEDUC-CE e SPAECE para fundamentar análises
    """
    return """
    CONTEXTO SEDUC-CE E SPAECE - FUNDAMENTAÇÃO DAS ANÁLISES:
    
    **SISTEMA PERMANENTE DE AVALIAÇÃO DA EDUCAÇÃO BÁSICA DO CEARÁ (SPAECE):**
    - Criado em 1992, é um dos sistemas de avaliação mais antigos e consolidados do Brasil
    - Avalia anualmente estudantes do 2º, 5º e 9º anos do Ensino Fundamental e 3ª série do Ensino Médio
    - Foco nas disciplinas de Língua Portuguesa e Matemática
    - Utiliza escalas de proficiência: 500 pontos (2º ano) e 1000 pontos (5º, 9º anos e 3ª série EM)
    - Padrões de Desempenho: Crítico, Intermediário, Adequado (5º e 9º anos)
    - Padrões de Desempenho 2º ano: Não Alfabetizado, Alfabetização Incompleta, Intermediário, Suficiente, Desejável
    
    **PROGRAMA DE ALFABETIZAÇÃO NA IDADE CERTA (PAIC):**
    - Implementado desde 2007, é referência nacional em alfabetização
    - Foco na alfabetização até o 2º ano do Ensino Fundamental
    - Estrutura: 5 eixos (Gestão Municipal, Gestão Escolar, Avaliação, Formação de Professores, Material Didático)
    - Resultado: Ceará saltou de 22º para 1º lugar no IDEB entre 2005-2017
    
    **POLÍTICAS EDUCACIONAIS DO CEARÁ:**
    - Bônus por Resultado: Sistema de premiação baseado em desempenho
    - Aprender Pra Valer: Programa de fortalecimento da aprendizagem
    - Mais Paic: Expansão do PAIC para o 3º ao 5º ano
    - Jovem de Futuro: Parceria com Instituto Unibanco para Ensino Médio
    
    **INDICADORES DE REFERÊNCIA DO CEARÁ:**
    - IDEB 2021: 4º lugar nacional (5º ano: 6,4; 9º ano: 5,1; EM: 4,2)
    - Taxa de Aprovação: 95,2% (5º ano), 92,8% (9º ano)
    - Taxa de Abandono: 0,8% (5º ano), 2,1% (9º ano)
    - Proficiência Média SPAECE 2022: 5º ano LP: 225,8; MAT: 230,1
    - Proficiência Média SPAECE 2022: 9º ano LP: 275,3; MAT: 280,7
    
    **BENEFÍCIOS DA ALTA PARTICIPAÇÃO NO SPAECE:**
    - **Recursos Financeiros:** Municípios com alta participação podem receber mais recursos do FUNDEB e programas federais
    - **Melhoria da Estrutura:** Escolas com boa participação são priorizadas em investimentos em infraestrutura
    - **Planos de Carreira:** Altas taxas de participação servem de subsídio para implementar planos de cargos e carreiras
    - **Aumento Salarial:** Professores de escolas com boa participação podem ter aumentos salariais baseados em resultados
    - **Programas Especiais:** Acesso a programas como PAIC, Mais Paic e outros baseados em indicadores de qualidade
    - **Reputação Educacional:** Municípios com alta participação ganham reconhecimento e atraem mais investimentos
    - **IDEB Elevado:** Participação alta contribui para melhor IDEB, resultando em mais recursos e prestígio
    - **Políticas Públicas:** Dados de alta participação fundamentam políticas educacionais e alocação de recursos
    
    **METAS E PADRÕES DE REFERÊNCIA:**
    - Meta IDEB 2024: 5º ano: 6,5; 9º ano: 5,2; EM: 4,3
    - Padrão Adequado SPAECE: 5º ano LP: ≥200; MAT: ≥225
    - Padrão Adequado SPAECE: 9º ano LP: ≥275; MAT: ≥300
    - Taxa de Participação Mínima: 80% (crítico), 90% (adequado), **100% (IDEAL)**
    - **Meta de Participação Ideal:** 100% - máxima participação garante dados representativos e traz benefícios
    
    **CARACTERÍSTICAS SOCIOECONÔMICAS DO CEARÁ:**
    - População: 9,2 milhões de habitantes
    - PIB per capita: R$ 15.847 (2021)
    - Índice de Desenvolvimento Humano: 0,754 (2010)
    - Taxa de Pobreza: 25,8% (2021)
    - 184 municípios, 20 CREDEs (Coordenadorias Regionais de Desenvolvimento da Educação)
    
    **FATORES DE SUCESSO EDUCACIONAL:**
    - Continuidade das políticas públicas (16 anos de PAIC)
    - Foco na alfabetização e anos iniciais
    - Sistema de avaliação permanente e diagnóstico
    - Formação continuada de professores
    - Material didático específico e contextualizado
    - Gestão baseada em resultados e evidências
    - Parceria Estado-Municípios (regime de colaboração)
    
    **DESAFIOS ATUAIS:**
    - Redução do abandono escolar no Ensino Médio
    - Melhoria da proficiência em Matemática
    - Equidade entre regiões e grupos sociais
    - Impacto da pandemia na aprendizagem
    - Formação de professores em áreas específicas
    - Infraestrutura escolar em municípios menores
    
    **FONTES OFICIAIS:**
    - Site SEDUC-CE: https://www.seduc.ce.gov.br/
    - Portal SPAECE: https://spaece.seduc.ce.gov.br/
    - Relatórios anuais de resultados SPAECE
    - Documentos do PAIC e programas correlatos
    - Estatísticas educacionais do INEP/MEC
    """

def obter_contexto_banner(nome_grafico):
    """
    Retorna o contexto específico do banner 'Como analisar este gráfico' para nortear a análise IA
    """
    contextos = {
        "Taxa de Participação": """
        **CONTEXTO TÉCNICO DO GRÁFICO DE PARTICIPAÇÃO:**
        - Tipo: Gauge (medidor circular) com escala de 0% a 100%
        - Cores: Verde (90-100%), Amarelo (80-89%), Vermelho (<80%)
        - Fórmula: Taxa de participação = (Alunos Efetivos ÷ Alunos Previstos) × 100
        - Interpretação: Ponteiro indica taxa atual, zonas coloridas mostram classificação
        - Significado: Percentual de alunos que efetivamente participaram da avaliação
        - **Meta Ideal:** 100% de participação para garantir dados representativos e trazer benefícios
        
        **FOQUE APENAS NESTE GRÁFICO DE PARTICIPAÇÃO:**
        - Analise exclusivamente os dados de taxa de participação apresentados
        - Não mencione outros gráficos (proficiência, habilidades, desempenho, etc.)
        - Concentre-se apenas nos dados de participação e seus benefícios
        
        BENEFÍCIOS DA ALTA PARTICIPAÇÃO:
        - Recursos financeiros para o município (FUNDEB, programas federais)
        - Melhoria da estrutura escolar (priorização em investimentos)
        - Subsídio para planos de cargos e carreiras dos profissionais
        - Aumento salarial baseado em resultados
        - Acesso a programas especiais (PAIC, Mais Paic)
        - Reputação educacional e reconhecimento
        - IDEB elevado e mais investimentos
        - Fundamentação para políticas públicas educacionais
        """,
        
        "Proficiência Média": """
        **CONTEXTO TÉCNICO DO GRÁFICO DE PROFICIÊNCIA:**
        - Tipo: Cards com métricas e banners coloridos
        - Escalas: 500 (2º ano) e 1000 (5º e 9º anos)
        - Cores: Verde (Adequado), Amarelo (Intermediário), Vermelho (Crítico)
        - Interpretação: Valores numéricos de proficiência por entidade
        - Significado: Nível de conhecimento dos estudantes em cada entidade
        
        **FOQUE APENAS NESTE GRÁFICO DE PROFICIÊNCIA:**
        - Analise exclusivamente os dados de proficiência média apresentados
        - Não mencione outros gráficos (participação, habilidades, desempenho, etc.)
        - Concentre-se apenas nos dados de proficiência e suas implicações pedagógicas
        """,
        
        "Distribuição por Desempenho": """
        **CONTEXTO TÉCNICO DO GRÁFICO DE DESEMPENHO:**
        - Tipo: Gráfico de barras empilhadas (stacked bar chart)
        - Eixo X: Entidades (Estado, CREDE, Município, Escola)
        - Eixo Y: Percentual de alunos (0% a 100%)
        - Barras: Divididas em 5 segmentos (Níveis 1-5)
        - Padrões por etapa:
          * 2º Ano: Não Alfabetizado → Alfabetização Incompleta → Intermediário → Suficiente → Desejável
          * 5º/9º Ano: Muito Crítico → Crítico → Intermediário → Adequado
        - Interpretação: Altura total = 100% dos alunos, segmentos = proporção por nível
        
        **FOQUE APENAS NESTE GRÁFICO DE DESEMPENHO:**
        - Analise exclusivamente os dados de distribuição por desempenho apresentados
        - Não mencione outros gráficos (participação, proficiência, habilidades, etc.)
        - Concentre-se apenas nos dados de desempenho e estratégias por nível
        """,
        
        "Taxa de Acerto por Habilidade": """
        CONTEXTO TÉCNICO DO GRÁFICO:
        - Tipo: Gráfico de barras agrupadas (grouped bar chart)
        - Eixo X: Código da Habilidade (identificador único)
        - Eixo Y: Taxa de acerto (0% a 100%)
        - Barras: Agrupadas por tipo de entidade (Ceará, CREDE, Município, Escola)
        - Interpretação: Altura da barra = taxa de acerto, cores = tipo de entidade
        - Significado: Percentual de questões corretas por habilidade específica
        - Hierarquia: Habilidades têm pré-requisitos - básicas são fundamentais para avançadas
        """,
        
        "Proficiência por Etnia": """
        CONTEXTO SOCIOLÓGICO E TÉCNICO DO GRÁFICO:
        - Tipo: Gráfico de barras agrupadas (grouped bar chart)
        - Eixo X: Grupos étnicos (Branca, Preta, Parda, Amarela, Indígena)
        - Eixo Y: Proficiência média (escalas 500 ou 1000)
        - Barras: Agrupadas por tipo de entidade
        - Interpretação: Altura da barra = proficiência média do grupo étnico
        - Significado: Nível de conhecimento por grupo étnico-racial
        
        PERSPECTIVA SOCIOLÓGICA - EQUIDADE EDUCACIONAL:
        - FOCO PRINCIPAL: Identificar e analisar desigualdades educacionais entre grupos étnicos
        - QUESTÃO CENTRAL: Como o sistema educacional reproduz ou combate desigualdades raciais?
        - INDICADORES DE EQUIDADE: Proximidade dos resultados entre grupos étnicos
        - ANÁLISE CRÍTICA: Fatores sociais, históricos e estruturais que influenciam o desempenho
        - CONTEXTO HISTÓRICO: Herança de exclusão e discriminação racial no Brasil
        - POLÍTICAS PÚBLICAS: Ações afirmativas e políticas de equidade racial
        - INTERSECCIONALIDADE: Como raça se cruza com classe, gênero e território
        """,
        
        "Proficiência por NSE": """
        CONTEXTO SOCIOLÓGICO E TÉCNICO DO GRÁFICO:
        - Tipo: Gráfico de barras agrupadas (grouped bar chart)
        - Eixo X: Níveis Socioeconômicos (A, B, C, D, E)
        - Eixo Y: Proficiência média (escalas 500 ou 1000)
        - Barras: Agrupadas por tipo de entidade
        - Interpretação: Altura da barra = proficiência média do NSE
        - Significado: Nível de conhecimento por nível socioeconômico
        
        PERSPECTIVA SOCIOLÓGICA - EQUIDADE EDUCACIONAL:
        - FOCO PRINCIPAL: Analisar como a origem socioeconômica impacta o desempenho educacional
        - QUESTÃO CENTRAL: Como o sistema educacional reproduz ou combate desigualdades de classe?
        - INDICADORES DE EQUIDADE: Redução das diferenças entre NSEs (A, B, C, D, E)
        - ANÁLISE CRÍTICA: Fatores estruturais que perpetuam desigualdades socioeconômicas
        - CONTEXTO HISTÓRICO: Herança de exclusão social e concentração de renda no Brasil
        - CAPITAL CULTURAL: Como recursos familiares influenciam o desempenho escolar
        - POLÍTICAS PÚBLICAS: Ações de democratização do acesso e qualidade educacional
        - MOBILIDADE SOCIAL: Educação como instrumento de transformação social
        """,
        
        "Proficiência por Sexo": """
        CONTEXTO SOCIOLÓGICO E TÉCNICO DO GRÁFICO:
        - Tipo: Gráfico de barras agrupadas (grouped bar chart)
        - Eixo X: Gêneros (Feminino, Masculino)
        - Eixo Y: Proficiência média (escalas 500 ou 1000)
        - Barras: Agrupadas por tipo de entidade
        - Interpretação: Altura da barra = proficiência média por gênero
        - Significado: Nível de conhecimento por gênero
        
        PERSPECTIVA SOCIOLÓGICA - EQUIDADE EDUCACIONAL:
        - FOCO PRINCIPAL: Analisar diferenças de desempenho entre gêneros na educação
        - QUESTÃO CENTRAL: Como o sistema educacional reproduz ou combate desigualdades de gênero?
        - INDICADORES DE EQUIDADE: Proximidade dos resultados entre gêneros
        - ANÁLISE CRÍTICA: Fatores sociais e culturais que influenciam o desempenho por gênero
        - CONTEXTO HISTÓRICO: Herança de desigualdades de gênero na sociedade brasileira
        - ESTEREÓTIPOS: Como expectativas sociais afetam o desempenho educacional
        - POLÍTICAS PÚBLICAS: Ações de promoção da equidade de gênero na educação
        - INTERSECCIONALIDADE: Como gênero se cruza com raça, classe e território
        - REPRESENTAÇÃO: Papel da representatividade e modelos de referência
        """
    }
    
    return contextos.get(nome_grafico, "")

# ==================== FUNÇÃO DE ANÁLISE COM GROQ ====================

def analisar_dataframe_com_groq(df, nome_grafico, contexto="", entidade_consultada="", df_concatenado=None):
    """
    Analisa um DataFrame usando a API da Groq e retorna insights considerando a hierarquia educacional
    Usa df_concatenado para identificar corretamente a entidade e sua hierarquia
    Inclui contexto do PDF de referência quando disponível
    """
    try:
        # Verificar se a API key está configurada
        if 'groq' not in st.secrets or 'api_key' not in st.secrets.groq:
            return "⚠️ API key da Groq não configurada no secrets.toml"
        
        api_key = st.secrets.groq.api_key
        if api_key == "gsk_your_groq_api_key_here":
            return "⚠️ Configure sua API key da Groq no arquivo secrets.toml"
        
        # Importar groq apenas quando necessário
        from groq import Groq
        
        # Inicializar cliente Groq
        client = Groq(api_key=api_key)
        
        # Determinar tipo de entidade e hierarquia
        tipo_entidade = "Desconhecida"
        nivel_hierarquico = ""
        entidades_superiores = []
        nome_entidade_consultada = entidade_consultada  # Usar código como fallback
        
        # Usar df_concatenado se disponível, senão usar df
        df_para_identificacao = df_concatenado if df_concatenado is not None and not df_concatenado.empty else df
        
        if not df_para_identificacao.empty:
            # Tentar obter nome da entidade consultada da coluna CD_ENTIDADE
            if 'CD_ENTIDADE' in df_para_identificacao.columns and not df_para_identificacao['CD_ENTIDADE'].isna().all():
                # Converter entidade_consultada para string para comparação
                entidade_consultada_str = str(entidade_consultada)
                # Buscar a linha que corresponde à entidade consultada
                entidade_filtrada = df_para_identificacao[df_para_identificacao['CD_ENTIDADE'].astype(str) == entidade_consultada_str]
                if not entidade_filtrada.empty:
                    # Tentar obter nome da entidade de diferentes colunas
                    if 'NM_ENTIDADE' in df_para_identificacao.columns and not entidade_filtrada['NM_ENTIDADE'].isna().iloc[0]:
                        nome_entidade_consultada = f"{entidade_consultada} - {entidade_filtrada['NM_ENTIDADE'].iloc[0]}"
                    elif 'DC_TIPO_ENTIDADE' in df_para_identificacao.columns and not entidade_filtrada['DC_TIPO_ENTIDADE'].isna().iloc[0]:
                        nome_entidade_consultada = f"{entidade_consultada} - {entidade_filtrada['DC_TIPO_ENTIDADE'].iloc[0]}"
            
            # Verificar colunas de tipo de entidade - usar DC_TIPO_ENTIDADE para identificar o tipo
            if 'DC_TIPO_ENTIDADE' in df_para_identificacao.columns and 'CD_ENTIDADE' in df_para_identificacao.columns:
                # Converter entidade_consultada para string para comparação
                entidade_consultada_str = str(entidade_consultada)
                # Buscar o tipo de entidade específico da entidade consultada
                entidade_filtrada = df_para_identificacao[df_para_identificacao['CD_ENTIDADE'].astype(str) == entidade_consultada_str]
                if not entidade_filtrada.empty:
                    dc_tipo_entidade = str(entidade_filtrada['DC_TIPO_ENTIDADE'].iloc[0]).upper()
                    # Mapear DC_TIPO_ENTIDADE para tipos de entidade baseado na descrição
                    if 'ESTADO' in dc_tipo_entidade or 'CEARÁ' in dc_tipo_entidade:
                        tipo_entidade = "Estado"
                        nivel_hierarquico = "Nível Estadual"
                    elif 'CREDE' in dc_tipo_entidade or 'REGIONAL' in dc_tipo_entidade:
                        tipo_entidade = "CREDE/Regional"
                        nivel_hierarquico = "Nível Regional"
                        entidades_superiores = ["Estado"]
                    elif 'MUNICÍPIO' in dc_tipo_entidade or 'MUNICIPIO' in dc_tipo_entidade:
                        tipo_entidade = "Município"
                        nivel_hierarquico = "Nível Municipal"
                        entidades_superiores = ["Estado", "CREDE/Regional"]
                    elif 'ESCOLA' in dc_tipo_entidade or 'EEIEF' in dc_tipo_entidade or 'EEM' in dc_tipo_entidade:
                        tipo_entidade = "Escola"
                        nivel_hierarquico = "Nível Escolar"
                        entidades_superiores = ["Estado", "CREDE/Regional", "Município"]
                    else:
                        # Fallback para TP_ENTIDADE se DC_TIPO_ENTIDADE não for reconhecido
                        if 'TP_ENTIDADE' in df.columns:
                            tp_entidade = entidade_filtrada['TP_ENTIDADE'].iloc[0]
                            if tp_entidade == 1:
                                tipo_entidade = "Estado"
                                nivel_hierarquico = "Nível Estadual"
                            elif tp_entidade == 2:
                                tipo_entidade = "CREDE/Regional"
                                nivel_hierarquico = "Nível Regional"
                                entidades_superiores = ["Estado"]
                            elif tp_entidade == 3:
                                tipo_entidade = "Município"
                                nivel_hierarquico = "Nível Municipal"
                                entidades_superiores = ["Estado", "CREDE/Regional"]
                            elif tp_entidade == 4:
                                tipo_entidade = "Escola"
                                nivel_hierarquico = "Nível Escolar"
                                entidades_superiores = ["Estado", "CREDE/Regional", "Município"]
            
            # Verificar se há dados de entidades superiores na hierarquia - usar dados da entidade consultada
            if 'CD_ENTIDADE' in df_para_identificacao.columns:
                # Converter entidade_consultada para string para comparação
                entidade_consultada_str = str(entidade_consultada)
                entidade_filtrada = df_para_identificacao[df_para_identificacao['CD_ENTIDADE'].astype(str) == entidade_consultada_str]
                if not entidade_filtrada.empty:
                    if 'NM_ESTADO' in df_para_identificacao.columns and not entidade_filtrada['NM_ESTADO'].isna().iloc[0]:
                        entidades_superiores.append(f"Estado: {entidade_filtrada['NM_ESTADO'].iloc[0]}")
                    if 'NM_REGIONAL' in df_para_identificacao.columns and not entidade_filtrada['NM_REGIONAL'].isna().iloc[0]:
                        entidades_superiores.append(f"CREDE: {entidade_filtrada['NM_REGIONAL'].iloc[0]}")
                    if 'NM_MUNICIPIO' in df_para_identificacao.columns and not entidade_filtrada['NM_MUNICIPIO'].isna().iloc[0]:
                        entidades_superiores.append(f"Município: {entidade_filtrada['NM_MUNICIPIO'].iloc[0]}")
        
        # Preparar dados para análise - limpeza inteligente de valores faltantes
        # Valores faltantes indicam que a coluna não tem registro para aquela linha específica
        # Para análise, manter linhas que tenham pelo menos alguns dados válidos
        df_limpo = df.copy()
        
        # Se o DataFrame está completamente vazio após dropna(), usar estratégia alternativa
        if df.dropna().empty and not df.empty:
            # Manter linhas que tenham pelo menos 50% das colunas com dados válidos
            threshold = len(df.columns) * 0.5
            df_limpo = df.dropna(thresh=threshold)
            
            # Se ainda estiver vazio, manter linhas com pelo menos 25% das colunas
            if df_limpo.empty:
                threshold = len(df.columns) * 0.25
                df_limpo = df.dropna(thresh=threshold)
                
            # Se ainda estiver vazio, usar o DataFrame original
            if df_limpo.empty:
                df_limpo = df
        else:
            df_limpo = df.dropna()
        
        df_info = {
            "nome_grafico": nome_grafico,
            "contexto": contexto,
            "entidade_consultada": nome_entidade_consultada,
            "tipo_entidade": tipo_entidade,
            "nivel_hierarquico": nivel_hierarquico,
            "entidades_superiores": entidades_superiores,
            "shape_original": df.shape,
            "shape_limpo": df_limpo.shape,
            "colunas": df.columns.tolist(),
            "tipos_dados": df.dtypes.to_dict(),
            "amostra_dados": df_limpo.head(10).to_dict('records') if not df_limpo.empty else [],
            "estatisticas": df_limpo.describe().to_dict() if not df_limpo.empty else {},
            "debug_info": {
                "entidade_consultada_original": entidade_consultada,
                "entidade_consultada_str": str(entidade_consultada),
                "cd_entidade_values": df_para_identificacao['CD_ENTIDADE'].unique().tolist()[:5] if 'CD_ENTIDADE' in df_para_identificacao.columns else "Coluna não encontrada",
                "dc_tipo_entidade_values": df_para_identificacao['DC_TIPO_ENTIDADE'].unique().tolist()[:5] if 'DC_TIPO_ENTIDADE' in df_para_identificacao.columns else "Coluna não encontrada",
                "nm_entidade_values": df_para_identificacao['NM_ENTIDADE'].unique().tolist()[:5] if 'NM_ENTIDADE' in df_para_identificacao.columns else "Coluna não encontrada",
                "entidade_encontrada": not df_para_identificacao.empty and 'CD_ENTIDADE' in df_para_identificacao.columns and str(entidade_consultada) in df_para_identificacao['CD_ENTIDADE'].astype(str).values,
                "dc_tipo_entidade_da_entidade": entidade_filtrada['DC_TIPO_ENTIDADE'].iloc[0] if not df_para_identificacao.empty and 'CD_ENTIDADE' in df_para_identificacao.columns and 'DC_TIPO_ENTIDADE' in df_para_identificacao.columns and not df_para_identificacao[df_para_identificacao['CD_ENTIDADE'].astype(str) == str(entidade_consultada)].empty else "Não encontrado",
                "nm_entidade_da_entidade": entidade_filtrada['NM_ENTIDADE'].iloc[0] if not df_para_identificacao.empty and 'CD_ENTIDADE' in df_para_identificacao.columns and 'NM_ENTIDADE' in df_para_identificacao.columns and not df_para_identificacao[df_para_identificacao['CD_ENTIDADE'].astype(str) == str(entidade_consultada)].empty else "Não encontrado"
            }
        }
        
        # Criar prompt para análise
        # Adicionar contexto dos documentos usando RAG se disponível
        contexto_documentos = ""
        if st.session_state.get('documentos_carregados', False) and st.session_state.get('dados_rag'):
            dados_rag = st.session_state['dados_rag']
            
            # Usar RAG para encontrar informações relevantes
            consulta_especifica = f"{nome_grafico} {contexto} {tipo_entidade}"
            informacoes_relevantes = buscar_informacoes_relevantes(consulta_especifica, dados_rag, top_k=3)
            
            # Adicionar informações das tabelas se disponíveis
            tabelas_contexto = ""
            if dados_rag.get('tabelas'):
                tabelas_contexto = "\n\nDADOS DAS TABELAS DO DCRC:\n"
                for i, tabela in enumerate(dados_rag['tabelas'][:2], 1):
                    if isinstance(tabela, dict) and 'conteudo' in tabela:
                        tabelas_contexto += f"Tabela {i}:\n{tabela['conteudo'][:500]}...\n\n"
                    else:
                        # Fallback para outras estruturas de tabela
                        conteudo_fallback = str(tabela)[:500] if tabela else "Sem conteúdo"
                        tabelas_contexto += f"Tabela {i}:\n{conteudo_fallback}...\n\n"
            
            # Adicionar seções importantes
            secoes_contexto = ""
            if dados_rag.get('secoes_importantes'):
                secoes_contexto = "\n\nSEÇÕES IMPORTANTES DO DCRC:\n"
                for secao, conteudos in dados_rag['secoes_importantes'].items():
                    if conteudos:
                        secoes_contexto += f"{secao.upper()}:\n{conteudos[0][:300]}...\n\n"
            
            # Contexto específico para habilidades
            contexto_habilidades = ""
            if 'habilidade' in nome_grafico.lower() or 'competência' in nome_grafico.lower():
                # Adicionar comparação específica com competências do BNCC/DCRC
                comparacao_competencias = comparar_habilidades_competencias(dados_rag, nome_grafico)
                
                # Adicionar análise de percursos de aprendizado
                analise_percursos = analisar_percursos_aprendizado(dados_rag, nome_grafico)
                
                # Adicionar ações específicas para escola
                acoes_escola = gerar_acoes_escola_baseadas_pdfs(dados_rag, nome_grafico, contexto)
                
                contexto_habilidades = f"""

        ===== ANÁLISE HIERÁRQUICA DE HABILIDADES: RELAÇÕES, DEPENDÊNCIAS E PERCURSOS POR NÍVEL EDUCACIONAL =====
        
        **ANÁLISE HIERÁRQUICA OBRIGATÓRIA - PERSPECTIVA POR NÍVEL EDUCACIONAL:**
        
        1. MAPEAMENTO HIERÁRQUICO DE RELAÇÕES E DEPENDÊNCIAS:
           - IDENTIFIQUE EXATAMENTE quais habilidades aparecem próximas nos dados ESPECÍFICOS do DataFrame
           - ANALISE se habilidades com desempenho similar estão relacionadas de forma CONCRETA conforme BNCC/DCRC
           - MAPEIE dependências ESPECÍFICAS: quais habilidades são pré-requisito para outras conforme os documentos
           - IDENTIFIQUE habilidades "gargalo" ESPECÍFICAS que bloqueiam o desenvolvimento de outras
           - EXPLIQUE por que certas habilidades têm padrões similares de forma ESPECÍFICA baseado nas relações dos documentos
           - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores
        
        2. PERCURSOS HIERÁRQUICOS DE APRENDIZADO ESTRUTURADOS:
           - DESENHE percursos de aprendizado ESPECÍFICOS: quais habilidades devem ser desenvolvidas primeiro
           - MAPEIE a hierarquia ESPECÍFICA: habilidades básicas → intermediárias → avançadas conforme BNCC/DCRC
           - IDENTIFIQUE pontos de convergência CONCRETOS onde múltiplas habilidades se encontram
           - ANALISE transferências de conhecimento ESPECÍFICAS entre componentes usando as relações dos documentos
           - SUGIRA sequências de ensino ESPECÍFICAS baseadas nas relações internas identificadas nos documentos
           - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores
        
        3. RELAÇÃO HIERÁRQUICA ENTRE COMPONENTES E COMPETÊNCIAS:
           - MAPEIE como habilidades de diferentes componentes se conectam de forma CONCRETA conforme BNCC/DCRC
           - IDENTIFIQUE competências ESPECÍFICAS que dependem de múltiplos componentes baseado nas competências específicas
           - RELACIONE cada habilidade com competências específicas do BNCC/DCRC de forma CONCRETA
           - ANALISE lacunas ESPECÍFICAS entre habilidades e competências esperadas conforme os documentos
           - SUGIRA abordagens interdisciplinares ESPECÍFICAS baseadas nas relações identificadas nos documentos
           - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores
        
        4. DESCRIÇÕES E METODOLOGIAS HIERÁRQUICAS:
           - USE as descrições ESPECÍFICAS do BNCC/DCRC para entender o que cada habilidade envolve
           - COMPARE descrições com desempenho real nos dados ESPECÍFICOS do DataFrame
           - IDENTIFIQUE habilidades mal compreendidas pelos estudantes de forma CONCRETA baseado nas descrições
           - MAPEIE metodologias ESPECÍFICAS sugeridas nos documentos para cada habilidade
           - SUGIRA reformulações pedagógicas ESPECÍFICAS baseadas nas descrições dos documentos
           - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores
        
        5. INTERVENÇÕES HIERÁRQUICAS SISTÊMICAS E MONITORAMENTO:
           - DESENHE planos de ação ESPECÍFICOS baseados nos percursos de aprendizado identificados
           - IDENTIFIQUE pontos de intervenção mais eficazes de forma CONCRETA baseado nas dependências
           - MAPEIE como melhorar uma habilidade impacta outras habilidades de forma ESPECÍFICA
           - SUGIRA intervenções sistêmicas ESPECÍFICAS baseadas nas relações identificadas nos documentos
           - MONITORE progresso considerando as interconexões de forma CONCRETA conforme BNCC/DCRC
           - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores
        
        6. REFERENCIAMENTO HIERÁRQUICO OBRIGATÓRIO:
           - REFERENCIE SEMPRE: "Conforme a BNCC", "Segundo o DCRC", "Baseado nas relações identificadas"
           - CITE competências específicas e objetivos de aprendizagem mencionados nos documentos de forma CONCRETA
           - REFERENCIE metodologias e recursos sugeridos nos documentos de forma ESPECÍFICA
           - IDENTIFIQUE campos de experiência e práticas de linguagem dos documentos de forma CONCRETA
           - DIFERENCIE entre informações dos documentos vs. análises genéricas de forma CLARA
           - SEJA ESPECÍFICO: evite generalizações, foque nos dados específicos da entidade
           - **CITE OBRIGATORIAMENTE BNCC E DCRC**: Sempre que possível, referencie tanto a BNCC quanto o DCRC como fontes principais das metodologias, competências e diretrizes curriculares
           - **PERSPECTIVA HIERÁRQUICA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores
        
        {comparacao_competencias}
        
        {analise_percursos}
        
        {acoes_escola}
        """
            
            # Contexto específico para proficiência
            contexto_proficiencia = ""
            if 'proficiência' in nome_grafico.lower() or 'desempenho' in nome_grafico.lower():
                # Adicionar ações específicas para escola
                acoes_escola_prof = gerar_acoes_escola_baseadas_pdfs(dados_rag, nome_grafico, contexto)
                
                contexto_proficiencia = f"""

        ===== CONTEXTO ESPECÍFICO PARA ANÁLISE DE PROFICIÊNCIA (DCRC + BNCC) =====
        
        FOQUE ESPECIALMENTE EM:
        1. RELAÇÃO COM COMPETÊNCIAS GERAIS DA BNCC
        2. ALINHAMENTO COM OBJETIVOS DE APRENDIZAGEM
        3. PROGRESSÃO CURRICULAR POR ETAPAS
        4. CAMPOS DE EXPERIÊNCIA E ÁREAS DE CONHECIMENTO
        5. EXPECTATIVAS DE APRENDIZAGEM POR ANO/SÉRIE
        
        USE AS INFORMAÇÕES DO DCRC E BNCC PARA:
        - Contextualizar níveis de proficiência com expectativas curriculares
        - Identificar lacunas entre desempenho e objetivos da BNCC
        - Sugerir intervenções alinhadas com competências específicas
        - Relacionar proficiência com campos de experiência
        - Considerar princípios e fundamentos da BNCC
        
        {acoes_escola_prof}
        """
            
            # Adicionar análise personalizada baseada nos dados específicos
            analise_personalizada = gerar_analise_personalizada(dados_rag, df_info, nome_grafico, contexto)
            
            # Adicionar ações específicas para escola baseadas no tipo de gráfico
            acoes_escola_geral = gerar_acoes_escola_baseadas_pdfs(dados_rag, nome_grafico, contexto)
            
            if informacoes_relevantes:
                contexto_rag = "\n\n".join([info['texto'] for info in informacoes_relevantes])
                contexto_documentos = f"""

        ===== CONTEXTO DOS DOCUMENTOS DCRC + BNCC (INFORMAÇÕES RELEVANTES) =====
        
        INFORMAÇÕES ESPECÍFICAS ENCONTRADAS:
        {contexto_rag[:2000]}
        
        {tabelas_contexto}
        
        {secoes_contexto}
        
        {contexto_habilidades}
        
        {contexto_proficiencia}
        
        {analise_personalizada}
        
        {acoes_escola_geral}
        
        INSTRUÇÃO CRÍTICA - ANÁLISE CIRÚRGICA FUNDAMENTADA NOS DOCUMENTOS BNCC E DCRC:
        
        **ANÁLISE HIERÁRQUICA OBRIGATÓRIA - PERSPECTIVA POR NÍVEL EDUCACIONAL:**
        
        **OBRIGATÓRIO**: Sua análise deve ser fundamentada EXCLUSIVAMENTE nas informações dos documentos BNCC e DCRC apresentados acima. Evite análises genéricas ou baseadas em conhecimento geral. **ANALISE PELA HIERARQUIA EDUCACIONAL**: cada entidade deve se ver no contexto dos níveis superiores (escola dentro do município, município dentro da regional, etc.).
        
        **REFERENCIAMENTO HIERÁRQUICO OBRIGATÓRIO**:
        1. **CITE EXPLICITAMENTE** quando usar informações do BNCC ("conforme a BNCC", "segundo a Base Nacional Comum Curricular")
        2. **REFERENCIE DIRETAMENTE** quando usar informações do DCRC ("conforme o DCRC", "segundo o Documento Curricular Referencial do Ceará")
        3. **IDENTIFIQUE A FONTE** de cada recomendação (BNCC ou DCRC) de forma CONCRETA
        4. **SEJA ESPECÍFICO**: evite generalizações, foque nos dados específicos da entidade
        5. **CITE OBRIGATORIAMENTE BNCC E DCRC**: Sempre que possível, referencie tanto a BNCC quanto o DCRC como fontes principais das metodologias, competências e diretrizes curriculares
        6. **ANALISE PELA HIERARQUIA**: Considere como a entidade se posiciona em relação aos níveis superiores e inferiores
        
        **FUNDAMENTAÇÃO CURRICULAR HIERÁRQUICA**:
        7. **COMPETÊNCIAS GERAIS**: Relacione com as 10 competências gerais da BNCC de forma CONCRETA
        8. **COMPETÊNCIAS ESPECÍFICAS**: Cite competências específicas das áreas de conhecimento de forma ESPECÍFICA
        9. **HABILIDADES**: Referencie habilidades específicas mencionadas nos documentos de forma CONCRETA
        10. **OBJETIVOS DE APRENDIZAGEM**: Use expectativas de aprendizagem dos documentos de forma ESPECÍFICA
        11. **CAMPOS DE EXPERIÊNCIA**: Relacione com campos de experiência da BNCC de forma CONCRETA
        12. **ÁREAS DE CONHECIMENTO**: Contextualize com áreas de conhecimento específicas de forma ESPECÍFICA
        13. **PRÁTICAS DE LINGUAGEM**: Aplique práticas de linguagem quando relevante de forma CONCRETA
        
        **ANÁLISE PEDAGÓGICA HIERÁRQUICA**:
        14. **METODOLOGIAS**: Use metodologias ESPECÍFICAS sugeridas no DCRC para intervenções
        15. **RECURSOS**: Sugira recursos ESPECÍFICOS baseados nas orientações dos documentos
        16. **AVALIAÇÃO**: Aplique princípios de avaliação ESPECÍFICOS mencionados nos documentos
        17. **PROGRESSÃO**: Considere progressão curricular ESPECÍFICA definida nos documentos
        18. **INTERVENÇÕES**: Baseie intervenções nas diretrizes curriculares de forma CONCRETA
        
        **ESTRUTURA DE RESPOSTA HIERÁRQUICA OBRIGATÓRIA**:
        - **Fundamentação Documental**: Cite trechos ESPECÍFICOS dos documentos
        - **Análise Curricular**: Relacione dados ESPECÍFICOS com competências e habilidades
        - **Recomendações Baseadas em Evidências**: Use metodologias ESPECÍFICAS dos documentos
        - **Ações Pedagógicas**: Específicas baseadas nas diretrizes curriculares de forma CONCRETA
        - **Indicadores de Progressão**: Alinhados com expectativas de aprendizagem ESPECÍFICAS
        - **PERSPECTIVA HIERÁRQUICA**: Analise como a entidade se posiciona em relação aos níveis superiores e inferiores
        """
            else:
                # Fallback para contexto geral se RAG não encontrar informações específicas
                contexto_documentos = f"""

        ===== CONTEXTO DOS DOCUMENTOS DCRC + BNCC (GERAL) =====
        
        {st.session_state['documentos_referencia'][:3000]}
        
        {tabelas_contexto}
        
        {secoes_contexto}
        
        {contexto_habilidades}
        
        {contexto_proficiencia}
        
        {analise_personalizada}
        
        {acoes_escola_geral}
        
        INSTRUÇÃO CRÍTICA: Use OBRIGATORIAMENTE estas informações do DCRC e BNCC para contextualizar suas análises e descrever ações específicas para a escola. PERSONALIZE baseando-se nos dados específicos da entidade. REFERENCIE explicitamente os PDFs nas análises.
        """

        prompt = f"""
        **ANÁLISE HIERÁRQUICA** dos dados educacionais do SPAECE (Sistema Permanente de Avaliação da Educação Básica do Ceará) considerando a HIERARQUIA EDUCACIONAL.

        **CONTEXTO HIERÁRQUICO ESPECÍFICO:**
        - Entidade Consultada: {entidade_consultada}
        - Tipo de Entidade: {tipo_entidade}
        - Nível Hierárquico: {nivel_hierarquico}
        - Entidades Superiores: {', '.join(entidades_superiores) if entidades_superiores else 'Nenhuma'}

        **INFORMAÇÕES ESPECÍFICAS DO GRÁFICO:**
        - Nome: {nome_grafico}
        - Contexto: {contexto}
        - Dimensões Originais: {df_info['shape_original'][0]} linhas x {df_info['shape_original'][1]} colunas
        - Dimensões Após Limpeza: {df_info['shape_limpo'][0]} linhas x {df_info['shape_limpo'][1]} colunas
        - Colunas: {', '.join(df_info['colunas'])}

        **INSTRUÇÃO HIERÁRQUICA OBRIGATÓRIA:**
        ANALISE PELA HIERARQUIA EDUCACIONAL: cada entidade deve se ver no contexto dos níveis superiores (escola dentro do município, município dentro da regional, etc.). Use informações dos documentos BNCC/DCRC de forma CONCRETA e ESPECÍFICA. **CITE OBRIGATORIAMENTE BNCC E DCRC** como fontes principais das metodologias, competências e diretrizes curriculares.
        
        **IMPORTANTE - FORMATAÇÃO DE NÚMEROS:**
        - **Números de alunos:** SEMPRE arredonde para números inteiros (ex: 150 alunos, não 150,5 alunos)
        - **Percentuais:** Use 1 casa decimal (ex: 85,3%)
        - **Proficiência:** Use números inteiros (ex: 250 pontos, não 250,7 pontos)
        - **Evite:** "meio aluno", "0,5 alunos" ou qualquer número decimal para quantidade de pessoas
        
        **FOCO EXCLUSIVO NO GRÁFICO ATUAL:**
        - **ANALISE APENAS** o gráfico "{nome_grafico}" apresentado acima
        - **NÃO MENCIONE** outros gráficos ou análises (participação, proficiência, habilidades, etc.)
        - **FOQUE EXCLUSIVAMENTE** nos dados específicos deste gráfico
        - **NÃO FAÇA** comparações com outros tipos de gráficos
        - **MANTENHA** o foco apenas nos dados e contexto deste gráfico específico
        
        **COMPARAÇÃO HIERÁRQUICA ESPECÍFICA:**
        - **COMPARE APENAS** entre os níveis: Estado, Regional (CREDE), Municipal e Escolar
        - **NÃO MENCIONE** comparações nacionais ou benchmarks nacionais
        - **FOQUE** na comparação entre os níveis hierárquicos do Ceará
        - **EVITE** palavras como "benchmarks" ou "padrões nacionais"
        - **CONCENTRE-SE** na análise comparativa entre os níveis do estado do Ceará

        {obter_contexto_banner(nome_grafico)}

        {obter_contexto_seduc_spaece()}

        **OBSERVAÇÃO IMPORTANTE:** Valores faltantes (NaN) foram tratados inteligentemente na análise. Quando possível, foram removidos completamente. Quando isso resultaria em dados insuficientes, foram mantidas linhas com pelo menos 50% ou 25% das colunas válidas, pois valores faltantes indicam que aquela coluna não possui registro para aquela linha específica na estrutura do DataFrame.

        DADOS DE AMOSTRA (após limpeza):
        {json.dumps(df_info['amostra_dados'], indent=2, default=str)}

        ESTATÍSTICAS DESCRITIVAS (após limpeza):
        {json.dumps(df_info['estatisticas'], indent=2, default=str)}

        INFORMAÇÕES DE DEBUG:
        {json.dumps(df_info['debug_info'], indent=2, default=str)}
        {contexto_documentos}

        INSTRUÇÕES ESPECÍFICAS PARA ANÁLISE PROFUNDA E DETALHADA:
        
        **ANÁLISE ESTATÍSTICA AVANÇADA:**
        1. CALCULE métricas estatísticas completas (média, mediana, moda, desvio padrão, variância, coeficiente de variação, assimetria, curtose)
        2. REALIZE análise de distribuição (normalidade, outliers, percentis 25, 50, 75, 90, 95)
        3. CALCULE intervalos de confiança e margens de erro quando aplicável
        4. IDENTIFIQUE correlações significativas entre variáveis e calcule coeficientes de correlação
        5. ANALISE variabilidade intra e inter-grupos com medidas de dispersão
        6. CALCULE índices de desigualdade (Gini, Theil, etc.) quando relevante
        
        **ANÁLISE COMPARATIVA DETALHADA:**
        7. COMPARE entre os níveis hierárquicos: Estado, Regional (CREDE), Municipal e Escolar
        8. ANALISE evolução temporal (se disponível) com tendências e sazonalidades
        9. IDENTIFIQUE posicionamento relativo entre entidades com rankings e percentis
        10. CALCULE gaps de desempenho específicos e oportunidades de melhoria quantificadas
        11. COMPARE contra metas educacionais estabelecidas e padrões de referência do Ceará
        
        **ANÁLISE DE SEGMENTAÇÃO E DISPERSÃO:**
        12. IDENTIFIQUE subgrupos com desempenho diferenciado e analise suas características
        13. ANALISE variabilidade intra e inter-grupos com medidas estatísticas precisas
        14. CALCULE índices de desigualdade e concentração quando aplicável
        15. IDENTIFIQUE fatores explicativos para as diferenças observadas
        16. MAPEIE distribuição espacial e temporal dos resultados
        
        **ANÁLISE DE CORRELAÇÕES E RELAÇÕES CAUSAIS:**
        17. IDENTIFIQUE correlações significativas entre variáveis com coeficientes precisos
        18. ANALISE relações de causa e efeito com evidências estatísticas
        19. IDENTIFIQUE fatores de influência, mediadores e moderadores
        20. SUGIRA hipóteses explicativas para os padrões observados com fundamentação
        21. ANALISE cadeias causais e efeitos indiretos
        
        **ANÁLISE DE EQUIDADE E JUSTIÇA EDUCACIONAL:**
        22. AVALIE distribuição justa de oportunidades e resultados com métricas específicas
        23. IDENTIFIQUE grupos em desvantagem educacional com evidências quantitativas
        24. ANALISE fatores de exclusão e discriminação com dados concretos
        25. SUGIRA políticas de equidade e inclusão baseadas em evidências
        26. CALCULE índices de equidade e justiça educacional
        
        **ANÁLISE SOCIOLÓGICA CRÍTICA (para ETNIA, NSE, SEXO):**
        27. PERSPECTIVA SOCIOLÓGICA: Analise através de lente crítica da sociologia da educação
        28. FOCO EM EQUIDADE: Identifique e analise desigualdades educacionais entre grupos
        29. CONTEXTO HISTÓRICO: Considere heranças de exclusão e discriminação no Brasil
        30. FATORES ESTRUTURAIS: Analise como sistemas sociais perpetuam desigualdades
        31. INTERSECCIONALIDADE: Como raça, classe e gênero se cruzam nas desigualdades
        32. POLÍTICAS PÚBLICAS: Sugira ações afirmativas e políticas de equidade
        33. MOBILIDADE SOCIAL: Como a educação pode transformar realidades sociais
        34. CAPITAL CULTURAL: Como recursos familiares influenciam o desempenho
        35. ESTEREÓTIPOS: Como expectativas sociais afetam diferentes grupos
        36. REPRESENTAÇÃO: Papel da representatividade e modelos de referência
        
        **ANÁLISE DE HABILIDADES E COMPETÊNCIAS:**
        37. MAPEIE hierarquia de habilidades e pré-requisitos com base no DCRC
        38. IDENTIFIQUE gaps de aprendizagem específicos e quantificados
        39. ANALISE sequência pedagógica ideal baseada em competências
        40. SUGIRA intervenções diferenciadas por habilidade com estratégias específicas
        41. RELACIONE habilidades com competências específicas do DCRC
        42. ANALISE interdependências entre habilidades e competências
        
        **ANÁLISE DE PROFICIÊNCIA E DESEMPENHO:**
        43. USE escalas de referência adequadas (500/1000) com interpretação precisa
        44. ANALISE distribuição por níveis de desempenho com percentuais específicos
        45. IDENTIFIQUE fatores que explicam a proficiência com evidências
        46. SUGIRA estratégias de melhoria por nível com ações específicas
        47. RELACIONE com competências gerais da BNCC
        48. ANALISE alinhamento com objetivos de aprendizagem
        
        **ANÁLISE CONTEXTUAL E SISTÊMICA:**
        49. CONSIDERE fatores socioeconômicos, geográficos e institucionais
        50. ANALISE impacto de políticas públicas e programas específicos
        51. IDENTIFIQUE recursos e condições necessárias para melhoria
        52. SUGIRA mudanças sistêmicas necessárias com fundamentação
        
        **RECOMENDAÇÕES ESTRATÉGICAS PRIORITÁRIAS:**
        53. PRIORIZE ações por impacto e viabilidade com matriz de priorização
        54. DEFINA metas específicas e mensuráveis com indicadores claros
        55. SUGIRA cronograma de implementação com marcos temporais
        56. IDENTIFIQUE recursos necessários com estimativas quantificadas
        57. FOQUE na entidade específica consultada ({nome_entidade_consultada})
        58. ADAPTE conselhos ao tipo de gestor (Secretário Estadual, Coordenador Regional, Secretário Municipal, Diretor Escolar)
        
        **INDICADORES DE MONITORAMENTO E AVALIAÇÃO:**
        59. DEFINA métricas de processo e resultado específicas
        60. ESTABELEÇA metas intermediárias e finais quantificadas
        61. SUGIRA frequência de monitoramento com cronograma
        62. IDENTIFIQUE sinais de alerta e sucesso com thresholds específicos
        
        **ESTRUTURA DA RESPOSTA DETALHADA:**
        63. RESUMO EXECUTIVO (4-5 parágrafos com insights principais e números específicos)
        64. ANÁLISE ESTATÍSTICA AVANÇADA (métricas detalhadas com cálculos)
        65. ANÁLISE COMPARATIVA E BENCHMARKING (posicionamento relativo quantificado)
        66. ANÁLISE DE SEGMENTAÇÃO E DISPERSÃO (subgrupos e variabilidade específica)
        67. ANÁLISE DE CORRELAÇÕES E RELAÇÕES CAUSAIS (fatores explicativos com evidências)
        68. ANÁLISE DE EQUIDADE E JUSTIÇA EDUCACIONAL (desigualdades e inclusão quantificadas)
        69. ANÁLISE DE HABILIDADES/COMPETÊNCIAS (se aplicável, com mapeamento detalhado)
        70. ANÁLISE DE PROFICIÊNCIA E DESEMPENHO (se aplicável, com escalas precisas)
        71. ANÁLISE CONTEXTUAL E SISTÊMICA (fatores externos com impacto quantificado)
        72. RECOMENDAÇÕES ESTRATÉGICAS PRIORITÁRIAS (ações específicas com cronograma)
        73. INDICADORES DE MONITORAMENTO E AVALIAÇÃO (métricas de sucesso específicas)
        74. CONCLUSÕES E PRÓXIMOS PASSOS (síntese e direcionamento claro)
        
        **FUNDAMENTAÇÃO SEDUC-CE E SPAECE:**
        75. USE OBRIGATORIAMENTE o contexto da SEDUC-CE e SPAECE fornecido acima
        76. COMPARE resultados com indicadores de referência do Ceará (IDEB, proficiência média, taxas)
        77. CONTEXTUALIZE análises com políticas educacionais do estado (PAIC, Mais Paic, Aprender Pra Valer)
        78. REFERENCIE padrões de desempenho específicos do SPAECE (escalas 500/1000, níveis)
        79. RELACIONE com fatores de sucesso educacional do Ceará identificados
        80. CONSIDERE características socioeconômicas específicas do estado
        81. IDENTIFIQUE alinhamento com metas e padrões de referência estaduais
        82. SUGIRA ações baseadas em programas e iniciativas já implementadas no Ceará
        83. CONTEXTUALIZE desafios atuais do sistema educacional cearense
        84. REFERENCIE fontes oficiais (sites SEDUC-CE e SPAECE) quando apropriado
        
        **REQUISITOS DE QUALIDADE:**
        85. SEJA EXTREMAMENTE específico e detalhado com números concretos
        86. USE dados concretos e cálculos precisos com fórmulas quando aplicável
        87. FORNEÇA insights acionáveis e estratégicos com fundamentação
        88. MANTENHA foco na melhoria educacional e equidade com evidências
        89. EVITE análises superficiais - seja profundo e analítico
        90. USE linguagem técnica apropriada mas acessível
        91. FORNEÇA evidências para todas as afirmações com dados específicos
        92. REFERENCIE explicitamente os PDFs quando aplicável
        93. FOQUE na entidade específica consultada, não em todas as entidades
        94. ADAPTE conselhos ao tipo de gestor e sua esfera de influência
        95. FUNDAMENTE análises com contexto específico do Ceará e SPAECE

        Responda em português brasileiro.
        """
        
        # Fazer chamada para a API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"Você é um consultor educacional especializado em análise de dados do SPAECE com mais de 15 anos de experiência. Seu papel é aconselhar especificamente o gestor da entidade consultada ({nome_entidade_consultada}) sobre ações práticas e viáveis dentro de sua esfera de influência. Considere que este gestor tem poder apenas sobre seu nível hierárquico ({nivel_hierarquico}) e não pode influenciar outros níveis da hierarquia educacional. Forneça análises PROFUNDAS, DETALHADAS e ESTRATÉGICAS com evidências quantitativas e qualitativas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        return "⚠️ Biblioteca groq não instalada. Execute: pip install groq"
    except Exception as e:
        return f"❌ Erro na análise: {str(e)}"

# Sistema de Autenticação - Carregar do secrets.toml
try:
    # Combinar todas as credenciais em um único dicionário
    PASSWORDS = {}
    ENTITY_NAMES = {}
    
    # Carregar senha mestra
    MASTER_PASSWORD = st.secrets.get("master", {}).get("password", "SPAECE2024")
    
    # Carregar usuários regionais
    if "xregionais" in st.secrets:
        PASSWORDS.update(st.secrets["xregionais"])
        for codigo in st.secrets["xregionais"].keys():
            ENTITY_NAMES[codigo] = f"Regional {codigo}"
    
    # Carregar usuários municipais
    if "xmunicipios" in st.secrets:
        PASSWORDS.update(st.secrets["xmunicipios"])
        # Mapear códigos municipais para nomes
        municipios_map = {
            "2301000": "AQUIRAZ",
            "2303709": "CAUCAIA", 
            "2304285": "EUSEBIO",
            "2304954": "GUAIUBA",
            "2306256": "ITAITINGA",
            "2307650": "MARACANAU",
            "2307700": "MARANGUAPE",
            "2309706": "PACATUBA"
        }
        for codigo in st.secrets["xmunicipios"].keys():
            ENTITY_NAMES[codigo] = municipios_map.get(codigo, f"Município {codigo}")
    
    # Carregar usuários de escolas
    if "xescolas" in st.secrets:
        PASSWORDS.update(st.secrets["xescolas"])
        # Mapear códigos de escolas para nomes (usando comentários do secrets.toml)
        escolas_map = {
            # AQUIRAZ
            "23061197": "ALOISIO BERNARDO DE CASTRO EMEF",
            "23061723": "ANTONIO DE BRITO LIMA EMEF",
            "23060956": "BATOQUE EMEF",
            "23564385": "CENTRO DE EDUCACAO E CIDADANIA MANUEL ASSUNCAO PIRES",
            "23061251": "CENTRO DE EDUCACAO E CIDADANIA MARIA DE CASTRO BERNARDO",
            "23061634": "CLARENCIO CRISOSTOMO DE FREITAS EMEF",
            "23061014": "CORREGO DA MINHOCA EMEF",
            "23061758": "DIONISIA GUERRA EMEF",
            "23061022": "ERNESTO GURGEL VALENTE EMEF",
            "23061618": "ESCOLA MUNICIPAL DE ENSINO FUNDAMENTAL TIA ALZIRA",
            "23262672": "FERDINANDO TANSI CENTRO EDUCACIONAL MUNICIPAL",
            "23061049": "FRANCISCA MONTEIRO DA SILVA EMEF",
            "23061057": "FRANCISCO DA SILVA SAMPAIO EMEF",
            "23061650": "FRANCISCO GOMES FARIAS EMEF CEL",
            "23061073": "GUILHERME JANJA EMEF",
            "23061081": "HENRIQUE GONCALVES DA JUSTA FILHO EMEF",
            "23061774": "ISIDORO DE SOUSA ASSUNCAO EMEF",
            "23061090": "JARBAS PASSARINHO MIN EMEF",
            "23061790": "JOAO JAIME GADELHA EMEF",
            "23061804": "JOAO PIRES CARDOSO EMEF",
            "23061111": "JOAQUIM DE SOUSA TAVARES EMEF",
            "23204150": "JOSE ALMIR DA SILVA EMEF",
            "23061413": "JOSE CAMARA DE ALMEIDA EMEF",
            "23061146": "JOSE FERREIRA DA COSTA EMEF",
            "23204141": "JOSE ISAAC SARAIVA DA CUNHA EMEF",
            "23061820": "JOSE RAIMUNDO DA COSTA EMEF",
            "23060999": "JOSE RODRIGUES MONTEIRO EMEF",
            "23061162": "JUSCELINO KUBITSCHEK EMEF",
            "23061847": "JUVENAL PEREIRA FACANHA EMEF",
            "23061189": "LAGOA DE CIMA EMEF",
            "23061855": "LAGOA DO MATO DE SERPA EMEF",
            "23248750": "LAIS SIDRIM TARGINO EMEF",
            "23176423": "LEOLINA BATISTA RAMOS EMEF",
            "23204125": "LUIZ EDUARDO STUDART GOMES EMEF",
            "23061278": "MARIA FACANHA DE SA EMEF",
            "23061294": "MARIA MARGARIDA RAMOS COELHO EMEF",
            "23061685": "MARIA SOARES DE FREITAS EMEF",
            "23061430": "PLACIDO CASTELO EMEF",
            "23060905": "RAIMUNDA DE FREITAS FACANHA CEI",
            "23061626": "RAIMUNDA FERREIRA DA SILVA EMEF",
            "23061480": "RAIMUNDO RAMOS DA COSTA EMEF",
            "23061448": "RITA PAULA DE BRITO EMEF",
            "23061596": "VILA PAGA EMEF",
            "23061910": "VINDINA ASSUNCAO DE AQUINO EMEF",
            # PACATUBA - Exemplo de algumas escolas
            "23264292": "JOAO PAULO SAMPAIO DE MENEZES EEIEF",
            "23083417": "ANA ALBUQUERQUE CAMPOS EEIEF",
            "23083433": "ANGELA COSTA CAMPOS EEF",
            "23083735": "CLOVIS DE CASTRO PEREIRA EEF",
            "23083492": "CRISPIANA DE ALBUQUERQUE EEF",
            "23083450": "DR CARLOS ALBERTO DE ALMEIDA PONTE EEF",
            "23083751": "FIRMINO DE ABREU LIMA EEIEF",
            "23182342": "GELIA DA SILVA CORREIA EEIEF",
            "23083778": "JARDIM BOM RETIRO EREIEF",
            "23326662": "JOANA VASCONCELOS DE OLIVEIRA EMTI",
            "23264020": "JOSE BATISTA DE OLIVEIRA EEIEF",
            "23083697": "MAJOR MANOEL ASSIS NEPOMUCENO EEIEF",
            "23267259": "MANOEL ROSENDO FREIRE EEF",
            "23083611": "MANUEL PONTES DE MEDEIROS EEIEF",
            "23083700": "MARIA DE SA RORIZ EEIEF",
            "23083808": "MARIA GUIOMAR BASTOS CAVALCANTE PROFESSORA EEIEF",
            "23083638": "MARIA MIRTES HOLANDA DO VALE PROF EEF",
            "23083719": "MARIA MOCINHA ROCHA SA EEIEF",
            "23083824": "NELLY DE LIMA E MELO EEIEF",
            "23083760": "OS HEROIS DO TIMBO EEIEF",
            "23083832": "PEDRO DE SA RORIZ EEIEF",
            "23083506": "RAIMUNDA DA CRUZ ALEXANDRE EREIEF",
            "23083689": "VICENTE FERRER DE LIMA EEIEF",
            "23190906": "WALNEY DO CARMO LOPES EEIEF"
        }
        for codigo in st.secrets["xescolas"].keys():
            ENTITY_NAMES[codigo] = escolas_map.get(codigo, f"Escola {codigo}")
    
    if not PASSWORDS:
        st.error("❌ Nenhuma credencial encontrada no secrets.toml!")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Erro ao carregar secrets.toml: {str(e)}")
    st.stop()

# Configuração da página
st.set_page_config(
    page_title="Painel CECOM 1 - Resultados do SPAECE 2024", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# CSS Global para Relatório Formal
st.markdown("""
    <style>
    /* Reset e configurações globais */
    .stContainer {
        padding: 0 !important;
    }
    
    /* Tema de relatório formal */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1200px;
        background: #fafafa;
        font-family: 'Arial', sans-serif;
    }
    
    /* Cards de métricas estilo relatório formal */
    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="stMetric"]:hover {
        border-color: #26a737;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: #26a737;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #26a737;
        margin-bottom: 8px;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 12px;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Espaçamento entre colunas */
    [data-testid="column"] {
        padding: 0 8px;
    }
    
    /* Botões estilo relatório formal */
    .stButton > button {
        background: #26a737;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: #26a737;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Selectbox estilo relatório formal */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #d1d5db;
        border-radius: 6px;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #2ca02c;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.1);
    }
    
    /* Expanders estilo relatório formal */
    .streamlit-expander {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .streamlit-expanderHeader {
        background: #f9fafb;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        color: #1f2937;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* DataFrames estilo relatório formal */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Dividers estilo relatório formal */
    .stDivider {
        background: #d1d5db;
        height: 1px;
        border: none;
        margin: 1.5rem 0;
    }
    
    /* Headers de seção estilo relatório formal */
    .report-header {
        background: #26a737;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 6px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Cards de informação estilo relatório formal */
    .report-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .report-card-header {
        background: #f9fafb;
        border-bottom: 2px solid #2ca02c;
        padding: 0.75rem 1rem;
        margin: -1.5rem -1.5rem 1rem -1.5rem;
        border-radius: 6px 6px 0 0;
        font-weight: 700;
        color: #1f2937;
    }
    
    /* Cores da paleta do gráfico de habilidades */
    .color-primary { color: #2ca02c; }
    .color-secondary { color: #f59c00; }
    .color-success { color: #2ca02c; }
    .color-danger { color: #d62728; }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #26a737;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #26a737;
    }
    
    /* Estilos para impressão */
    @media print {
        /* Configurações gerais da página */
        * {
            -webkit-print-color-adjust: exact !important;
            color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        
        /* Container principal - sem compressão */
        .main .block-container {
            padding: 0.5in !important;
            max-width: none !important;
            width: 100% !important;
            margin: 0 !important;
        }
        
        /* Ocultar sidebar */
        .stSidebar {
            display: none !important;
        }
        
        /* Ocultar elementos interativos */
        .stButton > button,
        .stDownloadButton,
        .stSelectbox,
        .stTextInput,
        .stButton,
        .stDownloadButton {
            display: none !important;
        }
        
        /* Expanders */
        .stExpander {
            border: 1px solid #ccc !important;
        }
        
        /* Quebras de página estratégicas */
        .page-break {
            page-break-before: always;
            break-before: page;
        }
        
        .page-break-before {
            page-break-before: always;
            break-before: page;
        }
        
        .page-break-after {
            page-break-after: always;
            break-after: page;
        }
        
        /* Evitar quebras de página desnecessárias */
        .stMarkdown {
            page-break-inside: avoid;
            break-inside: avoid;
        }
        
        /* Evitar quebras em elementos pequenos */
        .stDivider {
            page-break-after: avoid;
            break-after: avoid;
        }
        
        .avoid-break {
            break-inside: avoid;
            page-break-inside: avoid;
        }
        
        /* Headers sempre no topo da página */
        .report-header {
            break-inside: avoid;
            page-break-inside: avoid;
            break-after: avoid;
            page-break-after: avoid;
            margin: 0.5rem 0 !important;
            padding: 0.75rem 1rem !important;
        }
        
        /* Cards evitam quebra no meio */
        .report-card {
            break-inside: avoid;
            page-break-inside: avoid;
            margin-bottom: 1rem;
            padding: 1rem !important;
        }
        
        /* Métricas evitam quebra */
        div[data-testid="stMetric"] {
            break-inside: avoid;
            page-break-inside: avoid;
            margin: 0.25rem !important;
            padding: 0.75rem !important;
        }
        
        /* DataFrames podem quebrar se necessário */
        .stDataFrame {
            break-inside: auto;
            page-break-inside: auto;
            width: 100% !important;
            overflow: visible !important;
        }
        
        /* Gráficos evitam quebra */
        .stPlotlyChart {
            break-inside: avoid;
            page-break-inside: avoid;
            width: 100% !important;
        }
        
        /* Dividers evitam quebra */
        .stDivider {
            break-inside: avoid;
            page-break-inside: avoid;
            margin: 0.5rem 0 !important;
        }
        
        /* Seções que devem ficar na mesma página */
        .same-page-section {
            break-inside: avoid;
            page-break-inside: avoid;
            margin-bottom: 0.5rem !important;
        }
        
        /* Colunas responsivas */
        [data-testid="column"] {
            width: 100% !important;
            padding: 0.25rem !important;
        }
        
        /* Ajustar cores para impressão */
        .report-header {
            background: #2ca02c !important;
            color: white !important;
        }
        
        div[data-testid="stMetric"] {
            border: 2px solid #2ca02c !important;
        }
        
        div[data-testid="stMetricValue"] {
            color: #2ca02c !important;
        }
        
        /* Margens da página - reduzidas para melhor aproveitamento */
        @page {
            margin: 0.5in;
            size: A4 landscape;
        }
        
        /* Garantir que o conteúdo não seja comprimido */
        body {
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Ajustar tabelas para impressão */
        table {
            width: 100% !important;
            border-collapse: collapse !important;
        }
        
        /* Ajustar imagens e gráficos */
        img, svg {
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Corrigir compressão de colunas */
        .element-container {
            width: 100% !important;
            max-width: none !important;
        }
        
        /* Ajustar espaçamento geral */
        .stMarkdown {
            margin: 0.25rem 0 !important;
        }
        
        /* Corrigir largura de elementos */
        .stDataFrame > div {
            width: 100% !important;
            overflow: visible !important;
        }
        
        /* Ajustar botões ocultos */
        .stButton {
            display: none !important;
        }
        
        /* Corrigir layout de métricas */
        .metric-container {
            width: 100% !important;
            display: block !important;
        }
        
        /* Ajustar headers personalizados */
        .report-header {
            width: 100% !important;
            box-sizing: border-box !important;
        }
        
        /* Ajustar cards personalizados */
        .report-card {
            width: 100% !important;
            box-sizing: border-box !important;
        }
    }
    
    /* Estilo do botão Consultar com hover laranja */
    .stButton > button {
        background-color: #2ca02c !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #f59c00 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 127, 14, 0.3) !important;
    }
    
    /* Estilo customizado para botões de IA */
    .stButton > button[kind="primary"] {
        background-color: #dc3545 !important; /* Vermelho quando desativado */
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #c82333 !important; /* Vermelho mais escuro no hover */
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #28a745 !important; /* Verde quando ativado */
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #218838 !important; /* Verde mais escuro no hover */
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 6px rgba(255, 127, 14, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header estilo relatório formal com logos
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #26a737, #1e7e34, #155724);
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border: 3px solid #2ca02c;
        position: relative;
    ">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        ">
            <div style="flex: 0 0 auto; width: 180px;">
                <img src="data:image/png;base64,{}" style="max-width: 100%; height: auto; max-height: 120px;" />
            </div>
            <div style="flex: 1; text-align: center;">
                <h1 style="
                    color: white;
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin: 0;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                ">Resultados SPAECE 2024</h1>
                <p style="
                    color: rgba(255,255,255,0.9);
                    font-size: 1.1rem;
                    margin: 0.5rem 0 0 0;
                    font-weight: 500;
                ">Sistema Permanente de Avaliação da Educação Básica do Ceará</p>
            </div>
            <div style="flex: 0 0 auto; width: 180px;">
                <img src="data:image/png;base64,{}" style="max-width: 100%; height: auto; max-height: 120px;" />
            </div>
        </div>
        <div style="
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 2px solid rgba(255,255,255,0.3);
        ">
            <p style="
                color: rgba(255,255,255,0.8);
                font-size: 0.9rem;
                margin: 0;
                font-style: italic;
            ">Análise de Dados Educacionais - Relatório Executivo</p>
        </div>
    </div>
""".format(
    # Logo CECOM (lado esquerdo)
    base64.b64encode(open('logo_CECOM_branco2.png', 'rb').read()).decode(),
    # Logo CREDE (lado direito)
    base64.b64encode(open('logo_CREDE_branco2.png', 'rb').read()).decode()
), unsafe_allow_html=True)

# ==================== CONSTANTES ====================

# Códigos de tipos de entidade no sistema SPAECE
CODIGOS_ENTIDADE = {
    'ESTADO': '01',      # Estado do Ceará
    'CREDE': '02',       # Coordenadoria Regional de Desenvolvimento da Educação
    'MUNICIPIO': '11',   # Município
    'ESCOLA': '03'       # Escola individual
}

# ==================== FUNÇÕES DE API ====================

def consultar_api(agregado):
    """Consulta a API SPAECE com tratamento de erros aprimorado"""
    try:
        # Validar entrada
        if not agregado or not str(agregado).strip():
            st.error("❌ Código da Entidade não pode estar vazio")
            return None
            
        payload = criar_payload(
            indicadores=INDICADORES,
            agregado=str(agregado).strip(),
            filtros=[],
            nivel_abaixo="0"
        )
        
        response = requests.post(
            API_URL, 
            json=payload, 
            headers=HEADERS, 
            timeout=30
        )
        response.raise_for_status()
        
        # Verificar se a resposta contém dados válidos
        data = response.json()
        if not data:
            st.warning(f"⚠️ Nenhum dado retornado para o agregado {agregado}")
            return None
            
        return data
        
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Timeout ao consultar agregado {agregado}. Tente novamente.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"🌐 Erro de conexão. Verifique sua internet.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Erro HTTP {e.response.status_code}: {e.response.reason}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erro na requisição: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("❌ Erro ao decodificar resposta da API")
        return None
    except Exception as e:
        st.error(f"❌ Erro inesperado: {str(e)}")
        return None

# ==================== FUNÇÕES DE PROCESSAMENTO ====================

def processar_dados(data):
    """Processa dados da API e retorna DataFrame"""
    if not data:
        st.warning("⚠️ Nenhum dado fornecido para processamento")
        return None
        
    try:
        if isinstance(data, dict):
            if 'result' in data and data['result']:
                df = pd.DataFrame(data['result'])
            elif 'data' in data and data['data']:
                df = pd.DataFrame(data['data'])
            elif 'results' in data and data['results']:
                df = pd.DataFrame(data['results'])
            else:
                # Tentar normalizar dados aninhados
                df = pd.json_normalize(data)
        elif isinstance(data, list) and data:
            df = pd.DataFrame(data)
        else:
            st.warning("⚠️ Formato de dados não suportado")
            return None
            
        if df.empty:
            st.warning("⚠️ DataFrame vazio após processamento")
            return None
            
        return df
        
    except pd.errors.EmptyDataError:
        st.warning("⚠️ Dados vazios recebidos da API")
        return None
    except pd.errors.ParserError as e:
        st.error(f"❌ Erro ao processar dados: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Erro inesperado ao processar dados: {str(e)}")
        return None

def converter_para_numerico(df, colunas):
    """Converte colunas para formato numérico com tratamento robusto"""
    if df is None or df.empty:
        return df
        
    for col in colunas:
        if col in df.columns:
            try:
                # Substituir valores inválidos por NaN
                df[col] = df[col].replace(['-', 'N/A', 'n/a', '', 'NULL', 'null', 'None'], pd.NA).infer_objects(copy=False)
                
                # Limpar strings se for coluna de texto
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                    # Substituir strings vazias por NaN
                    df[col] = df[col].replace('', pd.NA).infer_objects(copy=False)
                
                # Converter para numérico
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            except Exception as e:
                st.warning(f"⚠️ Erro ao converter coluna '{col}': {str(e)}")
                continue
                
    return df

def extrair_agregados_hierarquia(df):
    """Extrai códigos de agregados da coluna DC_HIERARQUIA"""
    if 'DC_HIERARQUIA' not in df.columns:
        return []
    
    agregados = set()
    for valor in df['DC_HIERARQUIA'].dropna():
        if isinstance(valor, str):
            codigos = valor.split('/')
            agregados.update([cod.strip() for cod in codigos if cod.strip()])
    
    return sorted(list(agregados))

def obter_nome_entidade(df):
    """Obtém o nome da entidade da coluna NM_ENTIDADE"""
    if 'NM_ENTIDADE' in df.columns and not df['NM_ENTIDADE'].empty:
        nome = df['NM_ENTIDADE'].iloc[0]
        if pd.notna(nome):
            return str(nome)
    return None

def obter_tipo_entidade(df):
    """Obtém o tipo da entidade da coluna DC_TIPO_ENTIDADE"""
    if 'DC_TIPO_ENTIDADE' in df.columns and not df['DC_TIPO_ENTIDADE'].empty:
        tipo = df['DC_TIPO_ENTIDADE'].iloc[0]
        if pd.notna(tipo):
            return str(tipo).upper()
    return None

# ==================== FUNÇÕES DE VISUALIZAÇÃO ====================

def criar_card_entidade(titulo):
    """Cria um card HTML para exibir título de entidade"""
    return f"""
        <div style="
            border: 3px solid {COR_PRIMARIA};
            border-radius: 15px;
            padding: 20px 20px 5px 20px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            height: 120px;
            display: flex;
            flex-direction: column;
        ">
            <h3 style="
                text-align: center;
                color: #26a737;
                font-size: 1.4em;
                font-weight: bold;
                margin: 0 0 10px 0;
                padding-bottom: 10px;
                border-bottom: 3px solid {COR_PRIMARIA};
            ">{titulo}</h3>
        </div>
    """

def obter_proficiencia_media(df, codigo_tipo, coluna='Proficiência Média'):
    """Obtém a proficiência média para um tipo de entidade"""
    if df is None or df.empty:
        return None
    try:
        return df[df['Tipo de Entidade'].str.contains(codigo_tipo, case=False, na=False)][coluna].mean()
    except:
        return None

def aplicar_substituicoes(df):
    """Aplica substituições padronizadas nas colunas do DataFrame"""
    if df is None or df.empty:
        return df
        
    # Substituições para disciplina
    if 'VL_FILTRO_DISCIPLINA' in df.columns:
        df['VL_FILTRO_DISCIPLINA'] = df['VL_FILTRO_DISCIPLINA'].replace({
            'LP': 'Língua Portuguesa',
            'MT': 'Matemática'
        })
    
    # Substituições para rede
    if 'VL_FILTRO_REDE' in df.columns:
        df['VL_FILTRO_REDE'] = df['VL_FILTRO_REDE'].replace({
            'ESTADUAL': 'Estadual',
            'MUNICIPAL': 'Municipal',
            'PUBLICA': 'Pública',
            'PÚBLICA': 'Pública'
        })
    
    # Substituições para etapa
    if 'VL_FILTRO_ETAPA' in df.columns:
        df['VL_FILTRO_ETAPA'] = df['VL_FILTRO_ETAPA'].replace({
            'ENSINO FUNDAMENTAL DE 9 ANOS - 2º ANO': '2º Ano - Fundamental',
            'ENSINO FUNDAMENTAL DE 9 ANOS - 5º ANO': '5º Ano - Fundamental',
            'ENSINO FUNDAMENTAL DE 9 ANOS - 9º ANO': '9º Ano - Fundamental',
            'ENSINO MEDIO - 3ª SERIE': '3ª Série - Médio',
            'EJA DO ENSINO MEDIO - 3ª SÉRIE': '3ª Série - Médio EJA',
            'ENSINO MEDIO - 2ª SERIE': '2ª Série - Médio'
        })
    
    # Substituições para tipo de entidade
    if 'DC_TIPO_ENTIDADE' in df.columns:
        df['DC_TIPO_ENTIDADE'] = df['DC_TIPO_ENTIDADE'].replace({
            'ESTADO': 'Ceará',
            'REGIONAL': 'CREDE',
            'MUNICIPIO': 'Município',
            'ESCOLA': 'Escola'
        })
    
    return df

def criar_grafico_proficiencia(df, titulo, codigo_tipo, key_suffix):
    """
    Cria gráfico de proficiência média com cards e banners coloridos
    """
    if df.empty:
        st.warning("❌ Nenhum dado disponível para proficiência")
        return
    
    # Obter proficiência média
    prof_500 = obter_proficiencia_media(df, codigo_tipo, 'Proficiência Média 500')
    prof_1000 = obter_proficiencia_media(df, codigo_tipo, 'Proficiência Média 1000')
    
    # Criar cards de proficiência
    col1, col2 = st.columns(2)
    
    with col1:
        if not pd.isna(prof_500):
            st.metric("Proficiência 500", f"{prof_500:.0f}" if not pd.isna(prof_500) else "N/A", label_visibility="collapsed")
        
        with col2:
            if not pd.isna(prof_1000):
                st.metric("Proficiência 1000", f"{prof_1000:.0f}", label_visibility="collapsed")
            else:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #ff6b35, #f7931e);
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    margin-bottom: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                ">
                    📊 Escala 0-1000<br>
                    <span style="font-size: 12px;">Dados não disponíveis<br>para esta etapa</span>
                </div>
                """, unsafe_allow_html=True)

def criar_grafico_padrao_desempenho(df, titulo, codigo_tipo, key_suffix):
    """
    Cria gráfico de padrão de desempenho
    """
    if df.empty:
        st.warning("❌ Nenhum dado disponível para padrão de desempenho")
        return
    
    # Colunas de padrão de desempenho
    colunas_desempenho = [col for col in df.columns if 'Padrão' in col or 'Desempenho' in col]
    
    if not colunas_desempenho:
        st.warning("❌ Nenhuma coluna de padrão de desempenho encontrada")
        return
    
    # Criar gráfico de barras
    dados_grafico = []
    for col in colunas_desempenho:
        if col in df.columns:
            valor = df[col].iloc[0] if not df.empty else 0
            dados_grafico.append({
                'Categoria': col.replace('Padrão ', '').replace('Desempenho ', ''),
                'Valor': valor
            })
    
    if dados_grafico:
        df_grafico = pd.DataFrame(dados_grafico)
        fig = px.bar(df_grafico, x='Categoria', y='Valor', 
                    title=f"Distribuição por Padrão de Desempenho - {titulo}",
                    color='Valor',
                    color_continuous_scale=['#e06a0c', '#f59c00', '#26a737'])
        
        fig.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"desempenho_{key_suffix}")

def criar_grafico_habilidades(df, titulo, codigo_tipo, key_suffix):
    """
    Cria gráfico de habilidades
    """
    if df.empty:
        st.warning("❌ Nenhum dado disponível para habilidades")
        return
    
    # Colunas de habilidades
    colunas_habilidade = [col for col in df.columns if 'Habilidade' in col or 'Taxa' in col]
    
    if not colunas_habilidade:
        st.warning("❌ Nenhuma coluna de habilidade encontrada")
        return
    
    # Criar gráfico de barras
    dados_grafico = []
    for col in colunas_habilidade:
        if col in df.columns:
            valor = df[col].iloc[0] if not df.empty else 0
            dados_grafico.append({
                'Habilidade': col.replace('Taxa ', '').replace('Habilidade ', ''),
                'Taxa_Acerto': valor
            })
    
    if dados_grafico:
        df_grafico = pd.DataFrame(dados_grafico)
        fig = px.bar(df_grafico, x='Habilidade', y='Taxa_Acerto', 
                    title=f"Taxa de Acerto por Habilidade - {titulo}",
                    color='Taxa_Acerto',
                    color_continuous_scale=['#e06a0c', '#f59c00', '#26a737'])
        
        fig.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"habilidades_{key_suffix}")

def criar_gauge_participacao(df, titulo, codigo_tipo, key_suffix):
    """
    Cria um gauge de participação para um tipo específico de entidade dentro de um card
    
    Args:
        df: DataFrame com dados de participação
        titulo: Título do gauge
        codigo_tipo: Código para filtrar o tipo de entidade
        key_suffix: Sufixo para chave única do plotly
    """
    if df.empty:
        st.info("Sem dados de participação disponíveis")
        return
    
    # Filtrar dados do tipo específico
    dados_filtrados = df[df['Tipo de Entidade'].str.contains(codigo_tipo, case=False, na=False)]
    
    if dados_filtrados.empty:
        st.info(f"Sem dados de {titulo} para exibir")
        return
    
    # Encontrar a linha com o maior valor de participação
    idx_max_participacao = dados_filtrados['Participação'].idxmax()
    linha_max_participacao = dados_filtrados.loc[idx_max_participacao]
    
    # Pegar valores da linha com maior participação
    participacao_maxima = linha_max_participacao['Participação']
    total_previstos = linha_max_participacao['Alunos Previstos']
    total_efetivos = linha_max_participacao['Alunos Efetivos']
    
    # Card com altura fixa
    st.markdown(f"""
        <div style="
            border: 3px solid {COR_PRIMARIA};
            border-radius: 15px;
            padding: 20px 20px 5px 20px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            height: 120px;
            display: flex;
            flex-direction: column;
        ">
            <h3 style="
                text-align: center;
                color: #26a737;
                font-size: 1.4em;
                font-weight: bold;
                margin: 0 0 10px 0;
                padding-bottom: 10px;
                border-bottom: 3px solid {COR_PRIMARIA};
            ">{titulo}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Criar gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=participacao_maxima,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "", 'font': {'size': 14, 'color': COR_PRIMARIA}},
        number={'suffix': "%", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': COR_PRIMARIA},
            'steps': [
                {'range': [0, 80], 'color': "#ffcccc"},
                {'range': [80, 90], 'color': "#ffff99"},
                {'range': [90, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': participacao_maxima,
            }
        }
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=30, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{key_suffix}")
    
    # Métricas de alunos
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="👥 Previstos", 
            value=f"{int(total_previstos):,}".replace(',', 'X').replace('.', ',').replace('X', '.')
        )
    with col2:
        st.metric(
            label="✅ Efetivos", 
            value=f"{int(total_efetivos):,}".replace(',', 'X').replace('.', ',').replace('X', '.')
        )

# ==================== INICIALIZAÇÃO DO SESSION STATE ====================

if 'df_concatenado' not in st.session_state:
    st.session_state.df_concatenado = None
if 'agregado_consultado' not in st.session_state:
    st.session_state.agregado_consultado = None

# ==================== SISTEMA DE AUTENTICAÇÃO ====================

# Inicializar session state para autenticação
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_code' not in st.session_state:
    st.session_state.user_code = None

# Interface de Login
if not st.session_state.authenticated:
    st.markdown("### 🔐 Sistema de Autenticação SPAECE")
    
    # CSS customizado para botões laranja #ff7100 (aplicado globalmente)
    st.markdown("""
    <style>
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #e94f0e, #f59c00) !important;
        color: white !important;
        border: 3px solid #e94f0e !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        font-size: 1.1rem !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
        box-shadow: 0 3px 6px rgba(255, 113, 0, 0.4) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #e06a0c, #e94f0e) !important;
        border-color: #e06a0c !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 12px rgba(255, 113, 0, 0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            codigo = st.text_input("🏢 Código da Entidade", placeholder="Ex: 23, 230010, 230020...", help="Digite o código da entidade que deseja consultar, ou use a senha mestra para acessar todos os dados")
        
        with col2:
            senha = st.text_input("🔑 Senha", type="password", placeholder="Digite a senha da entidade")
        
        # Seleção de rede
        rede_selecionada = st.selectbox(
            "🏫 Rede de Ensino", 
            ["Selecione uma rede...", "Estadual", "Municipal"],
            index=0,
            help="Selecione a rede de ensino para filtrar os dados"
        )
        
        submitted = st.form_submit_button("🚀 Fazer Login e Consultar", type="secondary")
        
        if submitted:
            if not codigo or not senha:
                st.error("❌ Por favor, preencha todos os campos")
            elif rede_selecionada == "Selecione uma rede...":
                st.error("❌ Por favor, selecione uma rede de ensino")
            elif not codigo.isdigit():
                st.error("❌ O código da entidade deve conter apenas números")
            elif len(codigo) < 2:
                st.error("❌ O código da entidade deve ter pelo menos 2 dígitos")
            elif codigo not in PASSWORDS and senha != MASTER_PASSWORD:
                st.error("❌ Código da entidade não encontrado")
            elif PASSWORDS.get(codigo) != senha and senha != MASTER_PASSWORD:
                st.error("❌ Senha incorreta")
            else:
                # Login bem-sucedido
                st.session_state.authenticated = True
                st.session_state.user_code = codigo
                st.session_state.rede_selecionada_login = rede_selecionada
                
                # Verificar se é senha mestra e armazenar no session_state
                if senha == MASTER_PASSWORD:
                    st.session_state.master_access = True
                    st.success(f"✅ Login realizado com sucesso usando **SENHA MESTRA** para: **{ENTITY_NAMES.get(codigo, f'Entidade {codigo}')}** - Rede: **{rede_selecionada}**")
                else:
                    st.session_state.master_access = False
                    st.success(f"✅ Login realizado com sucesso para: **{ENTITY_NAMES.get(codigo, f'Entidade {codigo}')}** - Rede: **{rede_selecionada}**")
                st.rerun()
    
   
else:
    # Usuário autenticado - mostrar interface principal
    codigo = st.session_state.user_code
    nome_entidade = ENTITY_NAMES.get(codigo, f"Entidade {codigo}")
    
    # Botão de logout
    col1, col2 = st.columns([4, 1])
    with col1:
        rede_atual = st.session_state.get('rede_selecionada_login', 'Não definida')
        st.success(f"✅ Logado como: **{nome_entidade}** (Código: {codigo}) - Rede: **{rede_atual}**")
    with col2:
        # CSS para botão Sair laranja
        st.markdown("""
        <style>
        div[data-testid="stButton"] button[kind="secondary"] {
            background-color: #ff6b35 !important;
            color: white !important;
            border: none !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background-color: #e55a2b !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Sair", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.user_code = None
            st.session_state.agregado_consultado = None
            st.session_state.df_concatenado = None
            st.rerun()
    
    # Consulta automática usando o código do login
    agregado = codigo
    
    # Fazer consulta automaticamente
    if st.session_state.agregado_consultado != agregado:
        with st.spinner(f"🔄 Consultando dados da entidade {agregado}..."):
            # Lista para armazenar todos os dataframes
            lista_dfs = []
            
            # Consulta inicial
            data = consultar_api(agregado)
            if data:
                df = processar_dados(data)
                if df is not None:
                    # Adicionar coluna identificadora do agregado
                    df['AGREGADO_ORIGEM'] = agregado
                    lista_dfs.append(df)
                    
                    # LÓGICA ESPECIAL PARA SENHA MESTRA: Consultar dados mais amplos
                    if st.session_state.get('master_access', False):
                        st.info("🔑 **Acesso Administrativo:** Consultando dados ampliados...")
                        
                        # Para senha mestra, consultar também o estado completo (23)
                        if agregado != "23":
                            st.write("📊 Consultando dados do Estado do Ceará (23)...")
                            data_estado = consultar_api("23")
                            if data_estado:
                                df_estado = processar_dados(data_estado)
                                if df_estado is not None:
                                    df_estado['AGREGADO_ORIGEM'] = "23"
                                    lista_dfs.append(df_estado)
                        
                        # Consultar todas as CREDEs se for uma consulta estadual ou regional
                        if len(agregado) <= 2 or agregado == "23":
                            credes_para_consultar = ["230010", "230020", "230030", "230040", "230050", 
                                                    "230060", "230070", "230080", "230090", "230100", 
                                                    "230110", "230120", "230130", "230140", "230150", 
                                                    "230160", "230170", "230180", "230190", "230200"]
                            
                            for crede in credes_para_consultar:
                                st.write(f"📊 Consultando CREDE {crede}...")
                                data_crede = consultar_api(crede)
                                if data_crede:
                                    df_crede = processar_dados(data_crede)
                                    if df_crede is not None:
                                        df_crede['AGREGADO_ORIGEM'] = crede
                                        lista_dfs.append(df_crede)
                        
                        # Consultar municípios principais se for uma consulta estadual
                        if agregado == "23" or len(agregado) <= 2:
                            municipios_principais = ["2301000", "2303709", "2304285", "2304954", 
                                                   "2306256", "2307650", "2307700", "2309706"]
                            
                            for municipio in municipios_principais:
                                st.write(f"📊 Consultando Município {municipio}...")
                                data_municipio = consultar_api(municipio)
                                if data_municipio:
                                    df_municipio = processar_dados(data_municipio)
                                    if df_municipio is not None:
                                        df_municipio['AGREGADO_ORIGEM'] = municipio
                                        lista_dfs.append(df_municipio)
                        
                        # EXTRAIR AGREGADOS DA HIERARQUIA PARA SENHA MESTRA TAMBÉM
                        st.write("📊 Extraindo códigos da hierarquia...")
                        agregados_hierarquia = []
                        if len(agregado) > 2:
                            agregados_hierarquia = extrair_agregados_hierarquia(df)
                            agregados_hierarquia = [ag for ag in agregados_hierarquia if ag != agregado]
                        
                        # Consultar agregados da hierarquia para senha mestra
                        if agregados_hierarquia:
                            st.write(f"📊 Encontrados {len(agregados_hierarquia)} códigos na hierarquia")
                            for ag_hierarquia in agregados_hierarquia:
                                st.write(f"📊 Consultando hierarquia {ag_hierarquia}...")
                                data_hierarquia = consultar_api(ag_hierarquia)
                                if data_hierarquia:
                                    df_hierarquia = processar_dados(data_hierarquia)
                                    if df_hierarquia is not None:
                                        df_hierarquia['AGREGADO_ORIGEM'] = ag_hierarquia
                                        lista_dfs.append(df_hierarquia)
                    
                    # Extrair agregados da hierarquia (sem exibir) - apenas para usuários normais
                    else:
                        agregados_hierarquia = []
                        if len(agregado) > 2:
                            agregados_hierarquia = extrair_agregados_hierarquia(df)
                            agregados_hierarquia = [ag for ag in agregados_hierarquia if ag != agregado]
                        
                        # Consultar agregados da hierarquia silenciosamente
                        if agregados_hierarquia:
                            for ag_hierarquia in agregados_hierarquia:
                                data_hierarquia = consultar_api(ag_hierarquia)
                                if data_hierarquia:
                                    df_hierarquia = processar_dados(data_hierarquia)
                                    if df_hierarquia is not None:
                                        df_hierarquia['AGREGADO_ORIGEM'] = ag_hierarquia
                                        lista_dfs.append(df_hierarquia)
                    
                    # Criar dataframe concatenado (sem exibir)
                    if len(lista_dfs) == 1:
                        df_unico = lista_dfs[0].copy()
                        df_unico = aplicar_substituicoes(df_unico)
                        st.session_state.df_concatenado = df_unico
                        st.session_state.agregado_consultado = agregado
                    elif len(lista_dfs) > 1:
                        df_concatenado = pd.concat(lista_dfs, ignore_index=True)
                        df_concatenado = aplicar_substituicoes(df_concatenado)
                        st.session_state.df_concatenado = df_concatenado
                        st.session_state.agregado_consultado = agregado
                    
                    # Calcular total de registros
                    total_registros = sum(len(df) for df in lista_dfs)
                    st.success(f"✅ Dados carregados: {total_registros} registros (incluindo hierarquia)")
                    
                    # Opção de download do df_concatenado para usuários com senha mestra
                    if st.session_state.get('master_access', False):
                        st.info("🔑 **Acesso Administrativo:** Você pode baixar o dataset completo")
                        csv_data = st.session_state.df_concatenado.to_csv(index=False)
                        st.download_button(
                            label="📥 Baixar Dataset Completo (df_concatenado)",
                            data=csv_data,
                            file_name=f"spaece_dataset_completo_{codigo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_dataset_completo"
                        )
                else:
                    st.error("❌ Erro ao processar dados")
            else:
                st.warning("⚠️ Nenhum dado retornado pela API")
    
    # Verificar se há dados para exibir
    if st.session_state.df_concatenado is not None:
        # ==================== SEÇÃO DE ANÁLISE ====================
        df_concat = st.session_state.df_concatenado.copy()
        
        # Aplicar substituições para padronizar os nomes
        df_concat = aplicar_substituicoes(df_concat)
        
        
        # Header estilo relatório formal para análise dos dados
        st.markdown("""
        <div class="report-header" style="font-size: 2rem; text-align: left; background: linear-gradient(135deg, #2ca02c, #1e7e34, #155724);">
            📊 ANÁLISE DOS DADOS
        </div>


        """, unsafe_allow_html=True)
        
        # ==================== INFORMAÇÕES DA ENTIDADE ====================
        st.markdown("---")
        
        # Exibir informações da entidade consultada
        entidade_info = []
        
        # Verificar e adicionar nome da entidade principal
        if 'NM_ENTIDADE' in df_concat.columns and not df_concat.empty:
            entidade_nome = df_concat['NM_ENTIDADE'].iloc[0]
            if pd.notna(entidade_nome) and str(entidade_nome).strip():
                entidade_info.append(f"Entidade: {entidade_nome}")
        
        # Verificar e adicionar informações do município
        if 'NM_MUNICIPIO' in df_concat.columns and not df_concat.empty:
            municipio = df_concat['NM_MUNICIPIO'].iloc[0]
            if pd.notna(municipio) and str(municipio).strip():
                entidade_info.append(f"Município: {municipio}")
        
        # Verificar e adicionar informações da CREDE
        if 'NM_REGIONAL' in df_concat.columns and not df_concat.empty:
            crede = df_concat['NM_REGIONAL'].iloc[0]
            if pd.notna(crede) and str(crede).strip():
                entidade_info.append(f"CREDE: {crede}")
        
        # Verificar e adicionar informações do estado
        if 'NM_ESTADO' in df_concat.columns and not df_concat.empty:
            estado = df_concat['NM_ESTADO'].iloc[0]
            if pd.notna(estado) and str(estado).strip():
                entidade_info.append(f"Estado: {estado}")
        
        # Exibir as informações se existirem, senão mostrar Código da Entidade
        if entidade_info:
            # Card estilo relatório formal para informações da entidade
            # Criar o HTML com as informações da entidade
            entidade_html = "<br>".join(entidade_info)
            st.markdown(f"""
            <div class="report-card">
                <div class="report-card-header">
                    🏛️ INFORMAÇÕES DA ENTIDADE
                </div>
                <div style="
                    font-size: 1rem;
                    line-height: 1.8;
                    color: #374151;
                    font-weight: 500;
                ">
                    {entidade_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="report-card">
                <div class="report-card-header" style="border-bottom-color: #46ac33;">
                    🏛️ ENTIDADE CONSULTADA
                </div>
                <div style="
                    font-size: 1rem;
                    line-height: 1.8;
                    color: #4b5563;
                    font-weight: 500;
                ">
                    <strong>Código:</strong> {st.session_state.agregado_consultado}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # # ==================== ESTATÍSTICAS GERAIS ====================
    # with st.expander("📈 Estatísticas Gerais", expanded=False):
    #     col1, col2, col3, col4 = st.columns(4)
    #     with col1:
    #         st.metric("Total de Registros", f"{len(df_concat):,}".replace(',', '.'))
    #     with col2:
    #         if 'VL_FILTRO_DISCIPLINA' in df_concat.columns:
    #             st.metric("Componentes", df_concat['VL_FILTRO_DISCIPLINA'].nunique())
    #     with col3:
    #         if 'VL_FILTRO_ETAPA' in df_concat.columns:
    #             st.metric("Etapas", df_concat['VL_FILTRO_ETAPA'].nunique())
    #     with col4:
    #         if 'NM_ENTIDADE' in df_concat.columns:
    #             st.metric("Entidades", df_concat['NM_ENTIDADE'].nunique())
    
    # Filtrar por entidade se for consulta de nível estadual
    agregado_original = st.session_state.agregado_consultado
    if agregado_original and len(agregado_original) == 2:
        if 'CD_ENTIDADE' in df_concat.columns:
            df_concat = df_concat[df_concat['CD_ENTIDADE'] == agregado_original].copy()
            nome_estado = df_concat['NM_ENTIDADE'].iloc[0] if 'NM_ENTIDADE' in df_concat.columns and len(df_concat) > 0 else agregado_original
            st.info(f"🎯 Exibindo apenas dados da entidade: **{nome_estado}**")
    
    # Sidebar estilo relatório formal
    with st.sidebar:
        # Imagem do painel CECOM no topo do sidebar
        st.image("painel_cecom.png", width=300)
        
        # Card estilo relatório formal para informações da entidade no sidebar
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #26a737, #1e7e34, #155724);
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 2px solid #2ca02c;
        ">
            <h3 style="
                margin: 0;
                font-size: 1rem;
                font-weight: 700;
                text-align: center;
            ">🏛️ INFORMAÇÕES DA ENTIDADE</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Verificar e exibir informações da entidade
        sidebar_info = []
        
        # Verificar e adicionar nome da entidade principal
        if 'NM_ENTIDADE' in df_concat.columns and not df_concat.empty:
            entidade_nome = df_concat['NM_ENTIDADE'].iloc[0]
            if pd.notna(entidade_nome) and str(entidade_nome).strip():
                sidebar_info.append(f"Entidade: {entidade_nome}")
        
        # Verificar e adicionar informações do município
        if 'NM_MUNICIPIO' in df_concat.columns and not df_concat.empty:
            municipio = df_concat['NM_MUNICIPIO'].iloc[0]
            if pd.notna(municipio) and str(municipio).strip():
                sidebar_info.append(f"Município: {municipio}")
        
        # Verificar e adicionar informações da CREDE
        if 'NM_REGIONAL' in df_concat.columns and not df_concat.empty:
            crede = df_concat['NM_REGIONAL'].iloc[0]
            if pd.notna(crede) and str(crede).strip():
                sidebar_info.append(f"CREDE: {crede}")
        
        # Verificar e adicionar informações do estado
        if 'NM_ESTADO' in df_concat.columns and not df_concat.empty:
            estado = df_concat['NM_ESTADO'].iloc[0]
            if pd.notna(estado) and str(estado).strip():
                sidebar_info.append(f"Estado: {estado}")
        
        # Exibir as informações no sidebar com estilo relatório formal
        if sidebar_info:
            for info in sidebar_info:
                st.markdown(f"""
                <div style="
                    background: white;
                    padding: 0.6rem;
                    margin: 0.3rem 0;
                    border-radius: 4px;
                    border-left: 3px solid #2ca02c;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                    font-size: 0.8rem;
                    border: 1px solid #e5e7eb;
                ">{info}</div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: white;
                padding: 0.6rem;
                margin: 0.3rem 0;
                border-radius: 4px;
                border-left: 3px solid #6b7280;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                font-size: 0.8rem;
                border: 1px solid #e5e7eb;
            ">**Código:** {st.session_state.agregado_consultado}</div>
            """, unsafe_allow_html=True)
        
        # Header estilo relatório formal para filtros
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #26a737, #1e7e34, #155724);
            padding: 0.8rem;
            border-radius: 4px;
            margin: 1rem 0;
            text-align: center;
            border: 2px solid #2ca02c;
        ">
            <h3 style="
                margin: 0;
                color: white;
                font-size: 0.9rem;
                font-weight: 700;
            ">🔍 FILTROS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Filtro de Etapa - Dinâmico baseado na coluna VL_FILTRO_ETAPA
        if 'VL_FILTRO_ETAPA' in df_concat.columns:
            # Filtrar etapas apenas da entidade consultada
            df_etapas_entidade = df_concat.copy()
            
            # Se há uma entidade específica consultada, filtrar por ela
            agregado_original = st.session_state.agregado_consultado
            if agregado_original and 'CD_ENTIDADE' in df_concat.columns:
                df_etapas_entidade = df_etapas_entidade[df_etapas_entidade['CD_ENTIDADE'] == agregado_original]
            
            # Obter etapas únicas apenas da entidade consultada
            etapas_unicas = df_etapas_entidade['VL_FILTRO_ETAPA'].unique()
            etapas_unicas = [e for e in etapas_unicas if pd.notna(e)]  # Remove NaN
            etapas_unicas = [str(e).strip() for e in etapas_unicas if str(e).strip()]  # Remove espaços e valores vazios
            
            # Ordenar etapas de forma lógica (Educação Infantil -> Fundamental -> Médio)
            ordem_etapas = {
                'EDUCAÇÃO INFANTIL': 1,
                'EDUCAÇÃO INFANTIL - PRÉ-ESCOLA': 2,
                'ENSINO FUNDAMENTAL': 3,
                'ENSINO FUNDAMENTAL - ANOS INICIAIS': 4,
                'ENSINO FUNDAMENTAL - 1º ANO': 5,
                'ENSINO FUNDAMENTAL - 2º ANO': 6,
                'ENSINO FUNDAMENTAL - 3º ANO': 7,
                'ENSINO FUNDAMENTAL - 4º ANO': 8,
                'ENSINO FUNDAMENTAL - 5º ANO': 9,
                'ENSINO FUNDAMENTAL - ANOS FINAIS': 10,
                'ENSINO FUNDAMENTAL - 6º ANO': 11,
                'ENSINO FUNDAMENTAL - 7º ANO': 12,
                'ENSINO FUNDAMENTAL - 8º ANO': 13,
                'ENSINO FUNDAMENTAL - 9º ANO': 14,
                'ENSINO MÉDIO': 15,
                'ENSINO MÉDIO - 1ª SÉRIE': 16,
                'ENSINO MÉDIO - 2ª SÉRIE': 17,
                'ENSINO MÉDIO - 3ª SÉRIE': 18,
                'EJA': 19,
                'EJA DO ENSINO FUNDAMENTAL': 20,
                'EJA DO ENSINO MÉDIO': 21,
                'EJA DO ENSINO MÉDIO - 1ª SÉRIE': 22,
                'EJA DO ENSINO MÉDIO - 2ª SÉRIE': 23,
                'EJA DO ENSINO MÉDIO - 3ª SÉRIE': 24
            }
            
            # Ordenar etapas usando a ordem definida
            etapas_ordenadas = sorted(etapas_unicas, key=lambda x: ordem_etapas.get(x.upper(), 999))
            
            # Remover etapas específicas do seletor (se necessário)
            etapas_remover = [
                'ENSINO MÉDIO - 2ª SÉRIE',
                'ENSINO MÉDIO - 3ª SÉRIE', 
                'EJA DO ENSINO MÉDIO - 3ª SÉRIE'
            ]
            etapas_finais = [e for e in etapas_ordenadas if e.upper() not in [r.upper() for r in etapas_remover]]
            
            if len(etapas_finais) > 0:
                # Definir índice padrão (primeira etapa)
                default_index = 0
                
                etapa_selecionada = st.selectbox(
                    "📚 Selecione a Etapa de Ensino", 
                    etapas_finais, 
                    index=default_index,
                    key="etapa_selecionada",
                    help="Selecione uma etapa específica de ensino"
                )
                
                # Aplicar filtro
                if etapa_selecionada:
                    df_concat = df_concat[df_concat['VL_FILTRO_ETAPA'] == etapa_selecionada]
                    st.session_state.etapa_filtro_aplicado = etapa_selecionada
                    st.info(f"🔍 **Filtro aplicado:** {etapa_selecionada}")
            else:
                st.warning("⚠️ Nenhuma etapa encontrada na coluna VL_FILTRO_ETAPA")
        else:
            st.warning("⚠️ Coluna VL_FILTRO_ETAPA não encontrada nos dados")
        
        # Filtro de Disciplina - Dinâmico baseado na coluna VL_FILTRO_DISCIPLINA
        if 'VL_FILTRO_DISCIPLINA' in df_concat.columns:
            # Obter disciplinas únicas da coluna VL_FILTRO_DISCIPLINA
            disciplinas_unicas = df_concat['VL_FILTRO_DISCIPLINA'].unique()
            disciplinas_unicas = [d for d in disciplinas_unicas if pd.notna(d)]  # Remove NaN
            disciplinas_unicas = [str(d).strip() for d in disciplinas_unicas if str(d).strip()]  # Remove espaços e valores vazios
            
            # Ordenar disciplinas de forma lógica
            ordem_disciplinas = {
                'LÍNGUA PORTUGUESA': 1,
                'LÍNGUA PORTUGUESA - ESCRITA E LEITURA': 2,
                'MATEMÁTICA': 3,
                'CIÊNCIAS': 4,
                'HISTÓRIA': 5,
                'GEOGRAFIA': 6,
                'ARTES': 7,
                'EDUCAÇÃO FÍSICA': 8,
                'INGLÊS': 9,
                'ESPANHOL': 10,
                'FILOSOFIA': 11,
                'SOCIOLOGIA': 12,
                'FÍSICA': 13,
                'QUÍMICA': 14,
                'BIOLOGIA': 15
            }
            
            # Ordenar disciplinas usando a ordem definida
            disciplinas_ordenadas = sorted(disciplinas_unicas, key=lambda x: ordem_disciplinas.get(x.upper(), 999))
            
            if len(disciplinas_ordenadas) > 0:
                # Definir índice padrão (Língua Portuguesa se disponível, senão primeira disciplina)
                default_index = 0
                if "Língua Portuguesa" in disciplinas_ordenadas:
                    try:
                        default_index = disciplinas_ordenadas.index("Língua Portuguesa")
                    except ValueError:
                        default_index = 0
                elif "Língua Portuguesa - Escrita e Leitura" in disciplinas_ordenadas:
                    try:
                        default_index = disciplinas_ordenadas.index("Língua Portuguesa - Escrita e Leitura")
                    except ValueError:
                        default_index = 0
                
                disciplina_selecionada = st.selectbox(
                    "📖 Selecione a Disciplina", 
                    disciplinas_ordenadas, 
                    index=default_index,
                    key="disciplina_selecionada",
                    help="Selecione uma disciplina específica"
                )
                
                # Aplicar filtro
                if disciplina_selecionada:
                    df_concat = df_concat[df_concat['VL_FILTRO_DISCIPLINA'] == disciplina_selecionada]
                    st.session_state.disciplina_filtro_aplicado = disciplina_selecionada
                    st.info(f"🔍 **Filtro aplicado:** {disciplina_selecionada}")
            else:
                st.warning("⚠️ Nenhuma disciplina encontrada na coluna VL_FILTRO_DISCIPLINA")
        else:
            st.warning("⚠️ Coluna VL_FILTRO_DISCIPLINA não encontrada nos dados")
        
        # Aplicar filtro de rede selecionado no login
        rede_login = st.session_state.get('rede_selecionada_login', 'Estadual')
        if 'VL_FILTRO_REDE' in df_concat.columns:
            df_concat = df_concat[df_concat['VL_FILTRO_REDE'] == rede_login]
            st.info(f"🔍 **Rede selecionada no login:** {rede_login}")
        else:
            st.warning("⚠️ Coluna VL_FILTRO_REDE não encontrada nos dados")
        
        # Seção de controle da IA
        st.markdown("---")
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa, #e9ecef, #dee2e6);
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            margin: 1rem 0;
        ">
            <h4 style="color: #007bff; margin: 0 0 1rem 0;">🤖 Análise Inteligente com IA</h4>
            <p style="margin: 0 0 1rem 0; color: #6c757d;">
                Ative as análises inteligentes com IA para obter insights avançados dos dados. 
                <strong>Este processo carregará as bases de dados (DCRC e BNCC) e pode demorar alguns minutos</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Inicializar estado da IA se não existir (começar desligada)
        if 'ia_ativa' not in st.session_state:
            st.session_state.ia_ativa = False
        
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            if st.session_state.ia_ativa:
                if st.button("🤖 Desativar Análise IA", type="secondary", use_container_width=True, 
                            help="Clique para desativar as análises inteligentes com IA"):
                    st.session_state.ia_ativa = False
                    st.rerun()
            else:
                if st.button("🤖 Ativar Análise IA", type="primary", use_container_width=True,
                            help="Clique para ativar as análises inteligentes com IA"):
                    # Carregar arquivos Markdown quando ativar a IA
                    try:
                        with st.spinner("🔄 Carregando bases de dados..."):
                            # Carregar DCRC
                            with st.spinner("🔄 Carregando DCRC..."):
                                texto_dcrc = extrair_texto_md("dcrc.md")
                            
                            # Carregar BNCC
                            with st.spinner("🔄 Carregando BNCC..."):
                                texto_bncc = extrair_texto_md("bncc.md")
                            
                            if texto_dcrc and texto_bncc:
                                with st.spinner("🤖 Processando documentos com RAG..."):
                                    # Combinar textos dos dois arquivos Markdown
                                    texto_combinado = f"DCRC:\n{texto_dcrc}\n\nBNCC:\n{texto_bncc}"
                                    dados_rag = processar_md_com_rag(texto_combinado)
                                
                                if dados_rag:
                                    st.session_state.documentos_referencia = texto_combinado
                                    st.session_state.dados_rag = dados_rag
                                    st.session_state.documentos_carregados = True
                                    st.session_state.ia_ativa = True
                                    st.success("✅ IA ativada com sucesso! Bases carregadas e análises inteligentes habilitadas.")
                                    st.rerun()
                                else:
                                    st.error("❌ Erro ao processar os documentos. Tente novamente.")
                            else:
                                st.error("❌ Erro ao extrair texto dos PDFs. Verifique se os arquivos estão corretos.")
                                
                    except FileNotFoundError as e:
                        st.error(f"❌ Arquivo não encontrado: {e}")
                    except Exception as e:
                        st.error(f"❌ Erro ao carregar PDFs: {e}")
        
        # Mostrar status atual da IA
        if st.session_state.ia_ativa:
            st.markdown("""
            <div style="
                background: #e8f5e8;
                padding: 0.8rem;
                border-radius: 6px;
                border-left: 4px solid #28a745;
                margin: 1rem 0;
            ">
                <strong>✅ IA Ativa:</strong> Bases carregadas e análises inteligentes habilitadas
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: #f8f9fa;
                padding: 0.8rem;
                border-radius: 6px;
                border-left: 4px solid #6c757d;
                margin: 1rem 0;
            ">
                <strong>⏸️ IA Inativa:</strong> Análises inteligentes desabilitadas
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== TAXA DE PARTICIPAÇÃO ====================
    colunas_participacao = ['TP_ENTIDADE','NM_ENTIDADE','QT_ALUNO_PREVISTO','QT_ALUNO_EFETIVO', 
                           'TX_PARTICIPACAO', 'VL_FILTRO_DISCIPLINA','VL_FILTRO_ETAPA']
    
    if not df_concat.empty and all(col in df_concat.columns for col in colunas_participacao):
        df_participacao = df_concat[colunas_participacao].dropna().copy()
        df_participacao = df_participacao[df_participacao['VL_FILTRO_DISCIPLINA'] != 'Língua Portuguesa - Escrita e Leitura']
        df_participacao.columns = ['Tipo de Entidade', 'Entidade', 'Alunos Previstos', 'Alunos Efetivos', 
                                   'Participação', 'Componente Curricular', 'Etapa']
        
        # Só aplicar quebra de página se houver dados válidos após processamento
        if not df_participacao.empty:
            st.markdown("""
            <div class="report-header same-page-section" style="background: linear-gradient(135deg, #2ca02c, #1e7e34, #155724);">
                📊 TAXA DE PARTICIPAÇÃO
            </div>
            """, unsafe_allow_html=True)
            
            # Help para análise do gráfico
            with st.expander("ℹ️ Como analisar este gráfico", expanded=False):
                st.markdown("""
                **📊 Taxa de Participação - Informações Técnicas**
                
                **Construção do gráfico:**
                - **Tipo:** Gauge (medidor circular) com escala de 0% a 100%
                - **Cores:** Verde (90-100%), Amarelo (80-89%), Vermelho (<80%)
                - **Dados:** Taxa de participação = (Alunos Efetivos ÷ Alunos Previstos) × 100
                
                **O que representa:**
                - **Taxa de Participação:** Percentual de alunos que efetivamente participaram da avaliação
                - **Alunos Previstos:** Total de alunos matriculados que deveriam participar
                - **Alunos Efetivos:** Alunos que realmente fizeram a prova
                
                **Como ler:**
                - **Ponteiro:** Indica a taxa de participação atual
                - **Zonas coloridas:** Mostram faixas de classificação
                - **Valor numérico:** Taxa exata de participação
                """)
            
            # Converter para numérico
            df_participacao = converter_para_numerico(
                df_participacao, 
                ['Alunos Previstos', 'Alunos Efetivos', 'Participação']
            )
            
            # Gauges de participação
            col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
            
            with col1:
                criar_gauge_participacao(df_participacao, "Ceará", CODIGOS_ENTIDADE['ESTADO'], "ceara")
            
            with col2:
                criar_gauge_participacao(df_participacao, "CREDE", CODIGOS_ENTIDADE['CREDE'], "crede")
            
            with col3:
                criar_gauge_participacao(df_participacao, "Município", CODIGOS_ENTIDADE['MUNICIPIO'], "municipio")
            
            with col4:
                criar_gauge_participacao(df_participacao, "Escola", CODIGOS_ENTIDADE['ESCOLA'], "escola")
            
            # Download
            csv_part = df_participacao.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Baixar Dados de Participação",
                data=csv_part,
                file_name="participacao.csv",
                mime="text/csv",
                key="download_participacao"
            )
            
            # Análise com Groq
            with st.expander("🤖 Análise Inteligente - Taxa de Participação", expanded=False):
                if st.session_state.get('documentos_carregados', False) and st.session_state.get('ia_ativa', True):
                    st.error("⚠️ **Lembrete:** Esta análise é gerada por inteligência artificial e pode conter erros ou imprecisões. **Esta funcionalidade está em fase de testes.** Use sempre seu julgamento profissional para validar as informações.")
                    
                    # Criar chave única baseada nos filtros atuais
                    etapa_filtro = st.session_state.get('etapa_selecionada', 'Todas')
                    disciplina_filtro = st.session_state.get('disciplina_selecionada', 'Todas')
                    key_analise = f"analise_participacao_{etapa_filtro}_{disciplina_filtro}"
                    
                    if st.button("🔍 Analisar Dados com IA", key=key_analise):
                        with st.spinner("🤖 Analisando dados com IA..."):
                            analise = analisar_dataframe_com_groq(
                                df_participacao, 
                                "Taxa de Participação", 
                                "Análise da participação dos estudantes nas avaliações SPAECE. IMPORTANTE: O ideal é manter 100% de participação. Destaque como altas taxas de participação podem trazer recursos para o município, melhorar a estrutura da escola e servir de subsídio para implementar planos de cargos e carreiras e aumento de salário dos profissionais da educação, especialmente professores. Considere que participação alta é indicador de qualidade educacional e pode resultar em mais investimentos e melhorias estruturais.",
                                st.session_state.agregado_consultado,
                                st.session_state.df_concatenado
                            )
                            st.markdown(analise)
                elif not st.session_state.get('documentos_carregados', False):
                    st.warning("⚠️ **Análise IA indisponível:** Carregue as bases de dados (DCRC e BNCC) para ativar as análises inteligentes.")
                else:
                    st.warning("⚠️ **Análise IA desativada:** Use o botão no painel lateral para ativar as análises inteligentes.")
        else:
            st.info("Sem dados válidos de participação após processamento")
    else:
        st.info("Colunas necessárias não encontradas para exibir participação")
    
    # Espaçamento menor para manter na mesma página
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== PROFICIÊNCIA MÉDIA ====================
    st.markdown("""
    <div class="report-header same-page-section" style="background: linear-gradient(135deg, #2ca02c, #1e7e34, #155724);">
        📈 PROFICIÊNCIA MÉDIA
    </div>
    """, unsafe_allow_html=True)
    
    # Help para análise do gráfico
    with st.expander("ℹ️ Como analisar este gráfico", expanded=False):
        st.markdown("""
        **📈 Proficiência Média - Informações Técnicas**
        
        **Construção do gráfico:**
        - **Tipo:** Cards com métricas e banners coloridos
        - **Escalas:** Duas escalas diferentes (0-500 e 0-1000)
        - **Layout:** 4 colunas lado a lado (Estado, CREDE, Município, Escola)
        
        **O que representa:**
        - **Proficiência Média 500:** Pontuação média na escala de 0 a 500 pontos (2º e 5º anos)
        - **Proficiência Média 1000:** Pontuação média na escala de 0 a 1000 pontos (9º ano e EM)
        - **Banners:** Verde (escala 500) e Laranja (escala 1000)
        
        **Como ler:**
        - **Valores numéricos:** Pontuação média exata de cada entidade
        - **Banners coloridos:** Identificam qual escala está sendo mostrada
        - **Comparação:** Valores podem ser comparados entre as entidades
        """)
    
    colunas_proficiencia = ['TP_ENTIDADE','NM_ENTIDADE','AVG_PROFICIENCIA_E1','AVG_PROFICIENCIA_E2','VL_FILTRO_DISCIPLINA','VL_FILTRO_ETAPA']
    
    # Verificar se colunas de proficiência existem
    colunas_proficiencia_existentes = [col for col in colunas_proficiencia if col in df_concat.columns]
    
    # Tentar usar dados de proficiência específicos primeiro
    if not df_concat.empty and all(col in df_concat.columns for col in colunas_proficiencia):
        # Usar dropna apenas nas colunas essenciais, não nas de proficiência
        df_proficiencia = df_concat[colunas_proficiencia].dropna(
            subset=['TP_ENTIDADE', 'NM_ENTIDADE', 'VL_FILTRO_DISCIPLINA', 'VL_FILTRO_ETAPA']
        ).copy()
        
        df_proficiencia = df_proficiencia[df_proficiencia['VL_FILTRO_DISCIPLINA'] != 'Língua Portuguesa - Escrita e Leitura']
        
        df_proficiencia.columns = ['Tipo de Entidade', 'Entidade', 'Proficiência Média 500', 'Proficiência Média 1000', 'Componente Curricular', 'Etapa']
        
        # Converter para numérico
        df_proficiencia['Proficiência Média 500'] = pd.to_numeric(df_proficiencia['Proficiência Média 500'], errors='coerce')
        df_proficiencia['Proficiência Média 1000'] = pd.to_numeric(df_proficiencia['Proficiência Média 1000'], errors='coerce')
        
        # Debug: Verificar dados após processamento
        if len(df_proficiencia) > 0:
            pass  # Dados válidos encontrados
        else:
            st.warning("⚠️ DataFrame de proficiência está vazio após processamento")
    else:
        # Fallback: Criar dados de proficiência a partir dos dados disponíveis
        
        # Criar DataFrame básico com tipos de entidade
        if 'TP_ENTIDADE' in df_concat.columns and 'NM_ENTIDADE' in df_concat.columns:
            df_proficiencia = df_concat[['TP_ENTIDADE', 'NM_ENTIDADE']].copy()
            df_proficiencia.columns = ['Tipo de Entidade', 'Entidade']
            
            # Adicionar colunas de proficiência com valores padrão ou calculados
            df_proficiencia['Proficiência Média 500'] = 0.0
            df_proficiencia['Proficiência Média 1000'] = 0.0
            df_proficiencia['Componente Curricular'] = 'N/A'
            df_proficiencia['Etapa'] = 'N/A'
            
            # Tentar calcular proficiência média a partir de outras colunas se existirem
            colunas_proficiencia_alternativas = [col for col in df_concat.columns if 'PROFICIENCIA' in col.upper() or 'VL_' in col]
            
            if colunas_proficiencia_alternativas:
                # Para cada tipo de entidade, tentar calcular proficiência média
                for tipo_codigo in ['01', '02', '11', '03']:
                    dados_tipo = df_concat[df_concat['TP_ENTIDADE'] == tipo_codigo]
                    if len(dados_tipo) > 0:
                        # Tentar encontrar colunas de proficiência válidas
                        for col in colunas_proficiencia_alternativas:
                            if dados_tipo[col].notna().any():
                                try:
                                    prof_media = pd.to_numeric(dados_tipo[col], errors='coerce').mean()
                                    if not pd.isna(prof_media):
                                        # Atualizar dados de proficiência
                                        mask = df_proficiencia['Tipo de Entidade'] == tipo_codigo
                                        if 'E1' in col or '500' in col:
                                            df_proficiencia.loc[mask, 'Proficiência Média 500'] = prof_media
                                        elif 'E2' in col or '1000' in col:
                                            df_proficiencia.loc[mask, 'Proficiência Média 1000'] = prof_media
                                        break
                                except:
                                    continue
        else:
            st.error("❌ Não foi possível criar dados de proficiência")
            df_proficiencia = pd.DataFrame()
        
        # Debug mínimo: apenas mostrar se conseguiu criar dados
        if len(df_proficiencia) > 0:
            pass  # Dados criados com sucesso
        else:
            st.warning("⚠️ Não foi possível criar dados de proficiência")
    
    # Exibir gráficos de proficiência (para ambos os casos)
    if not df_proficiencia.empty:
        # DataFrame para exibição (sem a coluna Tipo de Entidade)
        df_proficiencia_display = df_proficiencia[['Entidade', 'Proficiência Média 500', 'Proficiência Média 1000', 'Componente Curricular', 'Etapa']].copy()
        
        col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
        
        # Cards de proficiência
        entidades = [
            ("Ceará", CODIGOS_ENTIDADE['ESTADO']),
            ("CREDE", CODIGOS_ENTIDADE['CREDE']),
            ("Município", CODIGOS_ENTIDADE['MUNICIPIO']),
            ("Escola", CODIGOS_ENTIDADE['ESCOLA'])
        ]
        
        for i, (nome, codigo) in enumerate(entidades):
            with [col1, col2, col3, col4][i]:
                st.markdown(criar_card_entidade(nome), unsafe_allow_html=True)
                proficiencia_500 = obter_proficiencia_media(df_proficiencia, codigo, 'Proficiência Média 500')
                proficiencia_1000 = obter_proficiencia_media(df_proficiencia, codigo, 'Proficiência Média 1000')
                
                # Espaço em cima dos banners
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Layout lado a lado para as duas escalas
                escala_col1, escala_col2 = st.columns(2)
                
                with escala_col1:
                    # Banner destacado para Escala 500
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {COR_PRIMARIA}, {COR_SUCESSO}); 
                               color: white; padding: 8px 10px; border-radius: 6px; 
                               text-align: center; font-weight: bold; font-size: 14px;
                               margin-bottom: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                        📊 Escala<br>0-500
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("Proficiência 500", f"{proficiencia_500:.0f}" if not pd.isna(proficiencia_500) else "N/A", label_visibility="collapsed")
                
                with escala_col2:
                    # Banner destacado para Escala 1000
                    if not pd.isna(proficiencia_1000):
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {COR_SECUNDARIA}, {COR_ACENTO}); 
                                   color: white; padding: 8px 10px; border-radius: 6px; 
                                   text-align: center; font-weight: bold; font-size: 14px;
                                   margin-bottom: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                            📊 Escala<br>0-1000
                        </div>
                        """, unsafe_allow_html=True)
                        st.metric("Proficiência 1000", f"{proficiencia_1000:.0f}", label_visibility="collapsed")
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {COR_SECUNDARIA}, {COR_ACENTO}); 
                                   color: white; padding: 8px 10px; border-radius: 6px; 
                                   text-align: center; font-weight: bold; font-size: 14px;
                                   margin-bottom: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                            📊 Escala<br>0-1000<br>
                            <span style="font-size: 11px;">Dados não disponíveis<br>para esta etapa</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        #Download
        csv_prof = df_proficiencia_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Baixar Dados de Proficiência",
            data=csv_prof,
            file_name="proficiencia.csv",
            mime="text/csv",
            key="download_proficiencia"
        )
        
        # Análise com Groq
        with st.expander("🤖 Análise Inteligente - Proficiência Média", expanded=False):
            if st.session_state.get('documentos_carregados', False) and st.session_state.get('ia_ativa', True):
                st.error("⚠️ **Lembrete:** Esta análise é gerada por inteligência artificial e pode conter erros ou imprecisões. **Esta funcionalidade está em fase de testes.** Use sempre seu julgamento profissional para validar as informações.")
                
                # Criar chave única baseada nos filtros atuais
                etapa_filtro = st.session_state.get('etapa_selecionada', 'Todas')
                disciplina_filtro = st.session_state.get('disciplina_selecionada', 'Todas')
                key_analise = f"analise_proficiencia_{etapa_filtro}_{disciplina_filtro}"
                
                if st.button("🔍 Analisar Dados com IA", key=key_analise):
                    with st.spinner("🤖 Analisando dados com IA..."):
                        analise = analisar_dataframe_com_groq(
                            df_proficiencia_display, 
                            "Proficiência Média", 
                            "Análise dos níveis de proficiência dos estudantes nas avaliações SPAECE (escalas 500 e 1000)",
                            st.session_state.agregado_consultado,
                            st.session_state.df_concatenado
                        )
                        st.markdown(analise)
            elif not st.session_state.get('documentos_carregados', False):
                st.warning("⚠️ **Análise IA indisponível:** Carregue as bases de dados (DCRC e BNCC) para ativar as análises inteligentes.")
            else:
                st.warning("⚠️ **Análise IA desativada:** Use o botão no painel lateral para ativar as análises inteligentes.")
    
    else:
        st.info("Sem dados válidos de proficiência após processamento")
    # ==================== DISTRIBUIÇÃO POR DESEMPENHO ====================
    colunas_desempenho = ['TP_ENTIDADE','DC_TIPO_ENTIDADE','NM_ENTIDADE','NU_N01_TRI_E1','NU_N02_TRI_E1','NU_N03_TRI_E1',
                         'NU_N04_TRI_E1','NU_N05_TRI_E1','TX_N01_TRI_E1', 'TX_N02_TRI_E1', 
                         'TX_N03_TRI_E1', 'TX_N04_TRI_E1', 'TX_N05_TRI_E1', 'VL_FILTRO_DISCIPLINA', 
                         'VL_FILTRO_ETAPA']
    
    if not df_concat.empty and all(col in df_concat.columns for col in colunas_desempenho):
        # Quebra de página antes da seção de desempenho (só se houver dados)
        st.markdown("""
        <div style="page-break-before: always; break-before: page;">
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="report-header" style="background: linear-gradient(135deg, {COR_SECUNDARIA}, #e67e22, #d35400);">
            📊 DISTRIBUIÇÃO POR PADRÃO DE DESEMPENHO
        </div>
        """, unsafe_allow_html=True)
        
        # Help para análise do gráfico
        with st.expander("ℹ️ Como analisar este gráfico", expanded=False):
            st.markdown("""
            **📊 Distribuição por Padrão de Desempenho - Informações Técnicas**
            
            **Construção do gráfico:**
            - **Tipo:** Gráfico de barras empilhadas (stacked bar chart)
            - **Eixo X:** Entidades (Estado, CREDE, Município, Escola)
            - **Eixo Y:** Percentual de alunos (0% a 100%)
            - **Barras:** Divididas em 5 segmentos (Níveis 1-5)
            
            **O que representa:**
            - **Nível 1-5:** Classificação dos estudantes por padrões de desempenho
            - **Percentual:** Proporção de alunos em cada nível
            - **Hover:** Mostra quantidade de alunos e percentual por nível
            
            **Padrões por etapa:**
            - **2º Ano:** Não Alfabetizado → Alfabetização Incompleta → Intermediário → Suficiente → Desejável
            - **5º/9º Ano:** Muito Crítico → Crítico → Intermediário → Adequado
            
            **Como ler:**
            - **Altura total:** 100% dos alunos avaliados
            - **Segmentos coloridos:** Proporção em cada nível de desempenho
            - **Hover:** Detalhes específicos de cada segmento
            """)
        
        # Usar dropna apenas nas colunas essenciais, não nas de desempenho
        df_desempenho = df_concat[colunas_desempenho].dropna(
            subset=['TP_ENTIDADE', 'DC_TIPO_ENTIDADE', 'NM_ENTIDADE', 'VL_FILTRO_DISCIPLINA', 'VL_FILTRO_ETAPA']
        ).copy()
        
        df_desempenho.columns = ['Tipo de Entidade', 'Tipo de Entidade Descrição', 'Entidade', 'Nível 1', 'Nível 2', 'Nível 3', 
                                'Nível 4', 'Nível 5', 'Taxa Nível 1', 'Taxa Nível 2', 
                                'Taxa Nível 3', 'Taxa Nível 4', 'Taxa Nível 5', 
                                'Componente Curricular', 'Etapa']
        
        # Definir ordem dos tipos de entidade e criar coluna de ordenação
        ordem_tipos = {'01': 1, '02': 2, '11': 3, '03': 4}
        df_desempenho['Ordem_Tipo'] = df_desempenho['Tipo de Entidade'].map(ordem_tipos)
        df_desempenho['Tipo de Entidade'] = pd.Categorical(
            df_desempenho['Tipo de Entidade'], 
            categories=['01', '02', '11', '03'], 
            ordered=True
        )
        
        # Ordenar o DataFrame principal pela ordem correta
        df_desempenho = df_desempenho.sort_values(['Ordem_Tipo', 'Entidade'])
        
        # Converter colunas de níveis para numérico
        df_desempenho = converter_para_numerico(
            df_desempenho, 
            ['Nível 1', 'Nível 2', 'Nível 3', 'Nível 4', 'Nível 5', 
             'Taxa Nível 1', 'Taxa Nível 2', 'Taxa Nível 3', 'Taxa Nível 4', 'Taxa Nível 5']
        )
        
        
        # Criar gráfico de barras para apresentar os níveis de desempenho
        st.subheader("Gráfico de Distribuição por Padrão de Desempenho")
        
        # Preparar dados para o gráfico
        df_grafico = df_desempenho.copy()
        
        # Ordenar por ordem numérica do tipo e depois por Entidade
        df_grafico = df_grafico.sort_values(['Ordem_Tipo', 'Entidade'])
        
        # Agrupar por tipo de entidade para manter as escolas individuais
        colunas_agregacao = {}
        for col in ['Taxa Nível 1', 'Taxa Nível 2', 'Taxa Nível 3', 'Taxa Nível 4', 'Taxa Nível 5']:
            if col in df_grafico.columns:
                colunas_agregacao[col] = 'mean'
        
        # Adicionar também as colunas de quantidade (Nível 1-5) para o hover
        for col in ['Nível 1', 'Nível 2', 'Nível 3', 'Nível 4', 'Nível 5']:
            if col in df_grafico.columns:
                colunas_agregacao[col] = 'sum'  # Somar as quantidades
        
        # Para escolas, manter individualmente; para outros tipos, agrupar por tipo
        df_escolas = df_grafico[df_grafico['Tipo de Entidade'] == '03'].copy()
        df_outros = df_grafico[df_grafico['Tipo de Entidade'] != '03'].copy()
        
        # Agregar outros tipos (Estado, CREDE, Município)
        if not df_outros.empty:
            df_outros_agregado = df_outros.groupby(['Tipo de Entidade', 'Tipo de Entidade Descrição', 'Ordem_Tipo'], observed=True).agg(colunas_agregacao).reset_index()
        else:
            df_outros_agregado = pd.DataFrame()
        
        # Manter escolas individuais (sem agregação)
        if not df_escolas.empty:
            df_escolas_agregado = df_escolas.groupby(['Tipo de Entidade', 'Tipo de Entidade Descrição', 'Ordem_Tipo', 'Entidade'], observed=True).agg(colunas_agregacao).reset_index()
        else:
            df_escolas_agregado = pd.DataFrame()
        
        # Combinar os DataFrames
        if not df_outros_agregado.empty and not df_escolas_agregado.empty:
            df_agregado = pd.concat([df_outros_agregado, df_escolas_agregado], ignore_index=True)
        elif not df_outros_agregado.empty:
            df_agregado = df_outros_agregado
        elif not df_escolas_agregado.empty:
            df_agregado = df_escolas_agregado
        else:
            df_agregado = pd.DataFrame()
        
        if len(df_agregado) > 0:
            # Ordenar o DataFrame agregado para manter a ordem no gráfico
            # Para escolas, ordenar por nome da escola; para outros, por tipo
            if 'Entidade' in df_agregado.columns:
                df_agregado = df_agregado.sort_values(['Ordem_Tipo', 'Tipo de Entidade Descrição', 'Entidade'])
            else:
                df_agregado = df_agregado.sort_values(['Ordem_Tipo', 'Tipo de Entidade Descrição'])
            
            # Detectar colunas de níveis disponíveis
            colunas_niveis_disponiveis = [col for col in df_agregado.columns if col.startswith('Taxa Nível')]
            
            # Transformar os dados para formato adequado para plotagem
            df_plot = pd.melt(
                df_agregado, 
                id_vars=['Tipo de Entidade', 'Tipo de Entidade Descrição', 'Ordem_Tipo'],
                value_vars=colunas_niveis_disponiveis,
                var_name='Padrão de Desempenho',
                value_name='Percentual'
            )
            
            # Ordenar o DataFrame plot para manter a ordem das entidades
            df_plot = df_plot.sort_values(['Ordem_Tipo', 'Tipo de Entidade Descrição'])
            
            # As colunas TX_* já são percentuais (0-100), não precisamos multiplicar por 100
            
            # Criar o gráfico de barras empilhadas usando go.Figure
            fig = go.Figure()
            
            # Detectar quantos níveis existem baseado na etapa
            # Usar df_desempenho que tem a coluna Etapa, não df_grafico que pode não ter
            etapa_atual = df_desempenho['Etapa'].iloc[0] if 'Etapa' in df_desempenho.columns and len(df_desempenho) > 0 else None
            
            if etapa_atual and ('2º Ano' in etapa_atual or '2º' in etapa_atual):
                # 2º ano tem APENAS 5 níveis
                niveis = ['Taxa Nível 1', 'Taxa Nível 2', 'Taxa Nível 3', 'Taxa Nível 4', 'Taxa Nível 5']
                cores = ['#e30513', '#fdc300', '#ffed00', '#cce4ce', '#1ca041']
                # Nomes para a legenda do 2º ano
                nomes_legenda = ['Não Alfabetizado', 'Alfabetização Incompleta', 'Intermediário', 'Suficiente', 'Desejável']
            elif etapa_atual and '5º Ano' in etapa_atual:
                # 5º ano tem 4 níveis
                niveis = ['Taxa Nível 1', 'Taxa Nível 2', 'Taxa Nível 3', 'Taxa Nível 4']
                cores = ['#e30513', '#fdc300', '#cce4ce', '#1ca041']
                # Nomes para a legenda do 5º ano
                nomes_legenda = ['Muito Crítico', 'Crítico', 'Intermediário', 'Adequado']
            elif etapa_atual and '9º Ano' in etapa_atual:
                # 9º ano tem 4 níveis
                niveis = ['Taxa Nível 1', 'Taxa Nível 2', 'Taxa Nível 3', 'Taxa Nível 4']
                cores = ['#e30513', '#fdc300', '#cce4ce', '#1ca041']
                # Nomes para a legenda do 9º ano
                nomes_legenda = ['Muito Crítico', 'Crítico', 'Intermediário', 'Adequado']
            else:
                # Fallback para outras etapas
                niveis = ['Taxa Nível 1', 'Taxa Nível 2', 'Taxa Nível 3', 'Taxa Nível 4']
                cores = ['#e30513', '#fdc300', '#cce4ce', '#1ca041']
                nomes_legenda = ['Muito Crítico', 'Crítico', 'Intermediário', 'Adequado']
            
            # Filtrar apenas os níveis que existem nos dados
            niveis_existentes = [nivel for nivel in niveis if nivel in df_agregado.columns]
            cores = cores[:len(niveis_existentes)]
            nomes_legenda_filtrados = nomes_legenda[:len(niveis_existentes)]
            
            # Adicionar uma barra para cada nível
            for i, nivel in enumerate(niveis_existentes):
                dados_nivel = df_plot[df_plot['Padrão de Desempenho'] == nivel]
                # Criar nome simplificado para o hover
                nome_nivel = nivel.replace('Taxa ', '')
                
                # Buscar os valores numéricos correspondentes (Nível 1, Nível 2, etc.)
                coluna_numerica = nome_nivel  # Nível 1, Nível 2, etc.
                
                # Criar dados para hover com quantidade de alunos
                hover_data = []
                for idx, row in dados_nivel.iterrows():
                    entidade_desc = row['Tipo de Entidade Descrição']
                    percentual = row['Percentual']
                    
                    # Para escolas, incluir nome da escola no hover
                    if row['Tipo de Entidade'] == '03' and 'Entidade' in row:
                        nome_display = f"{entidade_desc} - {row['Entidade']}"
                    else:
                        nome_display = entidade_desc
                    
                    # Buscar quantidade de alunos correspondente (coluna Nível 1, Nível 2, etc.)
                    # O percentual já vem da coluna Taxa Nível X, agora buscamos a quantidade da coluna Nível X
                    if 'Entidade' in df_agregado.columns:
                        # Para escolas, buscar por tipo + entidade
                        mask = (df_agregado['Tipo de Entidade Descrição'] == entidade_desc) & (df_agregado['Entidade'] == row.get('Entidade', ''))
                    else:
                        # Para outros tipos, buscar apenas por tipo
                        mask = df_agregado['Tipo de Entidade Descrição'] == entidade_desc
                    
                    quantidade_alunos = df_agregado[mask][coluna_numerica].iloc[0] if coluna_numerica in df_agregado.columns and mask.any() else 0
                    hover_data.append(f'<b>{nome_display}</b><br>Nível: {nomes_legenda_filtrados[i]}<br>Percentual: {percentual:.1f}%<br>Quantidade de Alunos: {quantidade_alunos:,.0f}')
                
                fig.add_trace(go.Bar(
                    name=nomes_legenda_filtrados[i],
                    x=dados_nivel['Percentual'],
                    y=dados_nivel.apply(lambda row: f"{row['Tipo de Entidade Descrição']} - {row['Entidade']}" if row['Tipo de Entidade'] == '03' and 'Entidade' in row else row['Tipo de Entidade Descrição'], axis=1),
                    orientation='h',
                    marker_color=cores[i],
                    customdata=hover_data,
                    hovertemplate='%{customdata}<extra></extra>',
                    text=dados_nivel['Percentual'].apply(lambda x: f'{x:.1f}%'),
                    textposition='inside',
                    textfont=dict(size=16, color='black')
                ))
            
            # Criar ordem específica das entidades baseada no tipo (01, 02, 11, 03)
            # Para escolas, usar nome da escola; para outros, usar tipo
            if 'Entidade' in df_plot.columns:
                df_ordenado = df_plot.sort_values(['Ordem_Tipo', 'Tipo de Entidade Descrição', 'Entidade'])
                # Para escolas, usar nome da escola; para outros, usar tipo
                ordem_manual = []
                for _, row in df_ordenado.iterrows():
                    if row['Tipo de Entidade'] == '03':  # Escola
                        nome_entidade = f"{row['Tipo de Entidade Descrição']} - {row['Entidade']}"
                    else:
                        nome_entidade = row['Tipo de Entidade Descrição']
                    if nome_entidade not in ordem_manual:
                        ordem_manual.append(nome_entidade)
            else:
                df_ordenado = df_plot.sort_values(['Ordem_Tipo', 'Tipo de Entidade Descrição'])
                ordem_manual = df_ordenado['Tipo de Entidade Descrição'].unique().tolist()
            
            # Configurar o layout para barras empilhadas
            fig.update_layout(
                barmode='stack',
                title=dict(
                    text='Distribuição por Padrão de Desempenho',
                    font=dict(size=24, family='Arial Black')
                ),
                xaxis_title=dict(
                    text='Percentual (%)',
                    font=dict(size=18)
                ),
                yaxis_title=dict(
                    text='Entidade',
                    font=dict(size=18)
                ),
                legend=dict(
                    title=dict(
                        text='Padrão de Desempenho',
                        font=dict(size=16)
                    ),
                    font=dict(size=16)
                ),
                height=500,
                yaxis=dict(
                    categoryorder='array', 
                    categoryarray=ordem_manual,
                    tickfont=dict(size=18)
                ),
                xaxis=dict(
                    tickfont=dict(size=18),
                    range=[0, 100]  # Forçar escala de 0 a 100%
                ),
                hoverlabel=dict(
                    font_size=16,
                    font_family="Arial"
                )
            )
            
            # Exibir o gráfico
            st.plotly_chart(fig, use_container_width=True)
            
            csv_desemp = df_desempenho.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Baixar Dados de Desempenho",
                data=csv_desemp,
                file_name="desempenho.csv",
                mime="text/csv",
                key="download_desempenho"
            )
            
            # Análise com Groq
            with st.expander("🤖 Análise Inteligente - Distribuição por Desempenho", expanded=False):
                if st.session_state.get('documentos_carregados', False) and st.session_state.get('ia_ativa', True):
                    st.error("⚠️ **Lembrete:** Esta análise é gerada por inteligência artificial e pode conter erros ou imprecisões. **Esta funcionalidade está em fase de testes.** Use sempre seu julgamento profissional para validar as informações.")
                    
                    # Criar chave única baseada nos filtros atuais
                    etapa_filtro = st.session_state.get('etapa_selecionada', 'Todas')
                    disciplina_filtro = st.session_state.get('disciplina_selecionada', 'Todas')
                    key_analise = f"analise_desempenho_{etapa_filtro}_{disciplina_filtro}"
                    
                    if st.button("🔍 Analisar Dados com IA", key=key_analise):
                        with st.spinner("🤖 Analisando dados com IA..."):
                            # Determinar os termos da legenda baseado na etapa
                            etapa_atual = df_desempenho['Etapa'].iloc[0] if 'Etapa' in df_desempenho.columns and len(df_desempenho) > 0 else None
                            if etapa_atual and '2º Ano' in etapa_atual:
                                termos_legenda = "Não Alfabetizado, Alfabetização Incompleta, Intermediário, Suficiente, Desejável"
                            elif etapa_atual and '5º Ano' in etapa_atual:
                                termos_legenda = "Muito Crítico, Crítico, Intermediário, Adequado"
                            elif etapa_atual and '9º Ano' in etapa_atual:
                                termos_legenda = "Muito Crítico, Crítico, Intermediário, Adequado"
                            else:
                                termos_legenda = "Muito Crítico, Crítico, Intermediário, Adequado"
                            
                            analise = analisar_dataframe_com_groq(
                                df_desempenho, 
                                "Distribuição por Desempenho", 
                                f"Análise da distribuição dos estudantes por padrões de desempenho ({termos_legenda})",
                                st.session_state.agregado_consultado,
                                st.session_state.df_concatenado
                            )
                            st.markdown(analise)
                elif not st.session_state.get('documentos_carregados', False):
                    st.warning("⚠️ **Análise IA indisponível:** Carregue as bases de dados (DCRC e BNCC) para ativar as análises inteligentes.")
                else:
                    st.warning("⚠️ **Análise IA desativada:** Use o botão no painel lateral para ativar as análises inteligentes.")
    else:
        st.info("Colunas necessárias não encontradas para exibir distribuição de desempenho")
    
    # ==================== TAXA DE ACERTO POR HABILIDADE ====================
    colunas_habilidade = ['TP_ENTIDADE','DC_TIPO_ENTIDADE','NM_ENTIDADE','VL_FILTRO_DISCIPLINA','VL_FILTRO_ETAPA',
                         'TX_ACERTO','DC_HABILIDADE','CD_HABILIDADE_MODELO_02']
    
    if not df_concat.empty and all(col in df_concat.columns for col in colunas_habilidade):
        # Quebra de página antes da seção de habilidades (só se houver dados)
        st.markdown("""
        <div style="page-break-before: always; break-before: page;">
        </div>
        """, unsafe_allow_html=True)
        
        # Só exibir o header se houver dados
        st.markdown("""
        <div class="report-header" style="background: linear-gradient(135deg, #d62728, #c82333, #a71e2a);">
            📚 TAXA DE ACERTO POR HABILIDADE
        </div>
        """, unsafe_allow_html=True)
        
        # Help para análise do gráfico
        with st.expander("ℹ️ Como analisar este gráfico", expanded=False):
            st.markdown("""
            **📚 Taxa de Acerto por Habilidade - Informações Técnicas**
            
            **Construção do gráfico:**
            - **Tipo:** Gráfico de barras agrupadas (grouped bar chart)
            - **Eixo X:** Código da Habilidade (identificador único)
            - **Eixo Y:** Taxa de acerto (0% a 100%)
            - **Barras:** Agrupadas por tipo de entidade (Ceará, CREDE, Município, Escola)
            
            **O que representa:**
            - **Taxa de Acerto:** Percentual de questões corretas por habilidade específica
            - **Código da Habilidade:** Identificador único de cada competência
            - **Habilidade:** Descrição da competência avaliada
            - **Comparação:** Entre tipos de entidade para cada habilidade
            
            **Como ler:**
            - **Altura da barra:** Taxa de acerto da habilidade para cada entidade
            - **Cores das barras:** Cada cor representa um tipo de entidade
            - **Agrupamento:** Barras lado a lado para comparar entidades
            - **Hover:** Mostra código, taxa de acerto e descrição da habilidade
            
            **Dados disponíveis:**
            - **Taxa de Acerto:** Percentual de questões corretas
            - **Código da Habilidade:** Identificador técnico
            - **Habilidade:** Descrição da competência
            - **Tipo de Entidade:** Ceará, CREDE, Município ou Escola
            """)
        
        df_habilidade = df_concat[colunas_habilidade].copy()
        
        df_habilidade.columns = ['Tipo de Entidade Código', 'Tipo de Entidade', 'Entidade', 'Componente Curricular', 'Etapa', 
                                'Taxa de Acerto', 'Habilidade', 'Código Habilidade']
        
        # Converter Taxa de Acerto para numérico
        df_habilidade['Taxa de Acerto'] = pd.to_numeric(df_habilidade['Taxa de Acerto'], errors='coerce')
        
        # Criar gráfico de barras para taxa de acerto por habilidade
        if not df_habilidade.empty:
            st.subheader("Gráfico de Taxa de Acerto por Habilidade")
            
            # Remover valores NaN e ordenar por taxa de acerto
            df_habilidade_grafico = df_habilidade.dropna(subset=['Taxa de Acerto']).copy()
            
            if not df_habilidade_grafico.empty:
                # Criar gráfico de barras agrupadas
                fig_habilidade = go.Figure()
                
                # Mapear tipos de entidade para nomes amigáveis
                mapa_tipos = {
                    '01': 'Ceará',
                    '02': 'CREDE',
                    '11': 'Município',
                    '03': 'Escola',
                    'Estado': 'Ceará',
                    'Regional': 'CREDE',
                    'Município': 'Município',
                    'Escola': 'Escola'
                }
                
                # Criar coluna com tipo de entidade amigável
                df_habilidade_grafico['Tipo Simplificado'] = df_habilidade_grafico['Tipo de Entidade Código'].map(mapa_tipos)
                if df_habilidade_grafico['Tipo Simplificado'].isna().any():
                    df_habilidade_grafico['Tipo Simplificado'] = df_habilidade_grafico['Tipo Simplificado'].fillna(
                        df_habilidade_grafico['Tipo de Entidade'].map(mapa_tipos)
                    )
                
                # Agrupar por tipo de entidade e código de habilidade, calculando a média
                df_agrupado = df_habilidade_grafico.groupby(['Tipo Simplificado', 'Código Habilidade', 'Habilidade']).agg({
                    'Taxa de Acerto': 'mean'
                }).reset_index()
                
                # Obter tipos de entidade únicos na ordem correta
                ordem_tipos = ['Ceará', 'CREDE', 'Município', 'Escola']
                tipos_disponiveis = [t for t in ordem_tipos if t in df_agrupado['Tipo Simplificado'].unique()]
                
                # Cores fixas para cada tipo de entidade
                cores_tipos = {
                    'Ceará': '#e94f0e',
                    'CREDE': '#f59c00',
                    'Município': '#26a737',
                    'Escola': '#2db39e'
                }
                
                # Adicionar uma barra para cada tipo de entidade
                for tipo in tipos_disponiveis:
                    df_tipo = df_agrupado[df_agrupado['Tipo Simplificado'] == tipo]
                    
                    if not df_tipo.empty:
                        # Ordenar por código da habilidade para manter consistência
                        df_tipo = df_tipo.sort_values('Código Habilidade')
                        
                        fig_habilidade.add_trace(go.Bar(
                            name=tipo,
                            x=df_tipo['Código Habilidade'],
                            y=df_tipo['Taxa de Acerto'],
                            text=df_tipo['Taxa de Acerto'].apply(lambda x: f'{x:.1f}'.replace('.', ',')) + '%',
                            textposition='auto',
                            textfont=dict(size=12, family='Arial', color='black'),
                            marker_color=cores_tipos.get(tipo, '#999999'),
                            hovertemplate=f'<b style="font-size:18px">{tipo}</b><br><span style="font-size:16px">Código: %{{x}}<br>Taxa de Acerto: %{{y:.1f}}%<br>Habilidade: %{{customdata}}</span><extra></extra>'.replace('.', ','),
                            customdata=df_tipo['Habilidade']
                        ))
                
                # Configurar o layout do gráfico
                fig_habilidade.update_layout(
                    title=dict(
                        text='Taxa de Acerto por Habilidade - Comparação entre Tipos de Entidade',
                        font=dict(size=18, family='Arial Black')
                    ),
                    xaxis_title=dict(
                        text='Código da Habilidade',
                        font=dict(size=14)
                    ),
                    yaxis_title=dict(
                        text='Taxa de Acerto (%)',
                        font=dict(size=14)
                    ),
                    legend=dict(
                        title=dict(
                            text='Tipo de Entidade',
                            font=dict(size=14)
                        ),
                        font=dict(size=14)
                    ),
                        font=dict(size=14),
                    height=400,
                    yaxis=dict(
                        range=[0, 100],
                        tickfont=dict(size=15)
                    ),
                    xaxis=dict(
                        tickfont=dict(size=15)
                    ),
                    barmode='group',
                    hoverlabel=dict(
                        font_size=18,
                        font_family="Arial"
                    )
                )
                
                # Exibir o gráfico
                st.plotly_chart(fig_habilidade, use_container_width=True)
            else:
                st.info("Sem dados válidos de taxa de acerto para criar o gráfico")
        else:
            st.info("Sem dados suficientes para criar o gráfico de habilidades")
        
        csv_hab = df_habilidade.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Baixar Dados de Habilidade",
            data=csv_hab,
            file_name="habilidade.csv",
            mime="text/csv",
            key="download_habilidade"
        )
        
        # Análise com Groq
        with st.expander("🤖 Análise Inteligente - Taxa de Acerto por Habilidade", expanded=False):
            if st.session_state.get('documentos_carregados', False) and st.session_state.get('ia_ativa', True):
                st.error("⚠️ **Lembrete:** Esta análise é gerada por inteligência artificial e pode conter erros ou imprecisões. **Esta funcionalidade está em fase de testes.** Use sempre seu julgamento profissional para validar as informações.")
                
                # Criar chave única baseada nos filtros atuais
                etapa_filtro = st.session_state.get('etapa_selecionada', 'Todas')
                disciplina_filtro = st.session_state.get('disciplina_selecionada', 'Todas')
                key_analise = f"analise_habilidade_{etapa_filtro}_{disciplina_filtro}"
                
                if st.button("🔍 Analisar Dados com IA", key=key_analise):
                    with st.spinner("🤖 Analisando dados com IA..."):
                        analise = analisar_dataframe_com_groq(
                            df_habilidade, 
                            "Taxa de Acerto por Habilidade", 
                            "Análise das habilidades específicas dos estudantes nas avaliações SPAECE. IMPORTANTE: Considere que as habilidades têm hierarquia de pré-requisitos - algumas são mais básicas e fundamentais que outras. Foque sempre em fortalecer as habilidades mais basilares primeiro, pois elas são pré-requisito para o desenvolvimento das demais. Identifique quais habilidades básicas precisam de mais atenção e como elas impactam o desenvolvimento das habilidades mais avançadas.",
                            st.session_state.agregado_consultado,
                            st.session_state.df_concatenado
                        )
                        st.markdown(analise)
            elif not st.session_state.get('documentos_carregados', False):
                st.warning("⚠️ **Análise IA indisponível:** Carregue as bases de dados (DCRC e BNCC) para ativar as análises inteligentes.")
            else:
                st.warning("⚠️ **Análise IA desativada:** Use o botão no painel lateral para ativar as análises inteligentes.")
    else:
        st.info("Colunas necessárias não encontradas para exibir taxa de acerto por habilidade")
    
    # Quebra de página antes da seção de etnia (removida para evitar páginas vazias)
    # st.markdown("""
    # <div style="page-break-before: always; break-before: page;">
    # </div>
    # """, unsafe_allow_html=True)
    
    # ==================== DADOS CONTEXTUAIS - ETNIA ====================
    colunas_etnia = ['TP_ENTIDADE','DC_TIPO_ENTIDADE','NM_ENTIDADE', 'VL_FILTRO_DISCIPLINA', 'VL_FILTRO_ETAPA', 'VL_PRETA',
                    'VL_BRANCA', 'VL_PARDA', 'VL_AMARELA', 'VL_INDIGENA','TX_PRETA','TX_BRANCA',
                    'TX_PARDA','TX_AMARELA','TX_INDIGENA','NU_PRETA','NU_BRANCA','NU_PARDA',
                    'NU_AMARELA','NU_INDIGENA']
    colunas_etnia_disponiveis = [col for col in colunas_etnia if col in df_concat.columns]
    
    if len(colunas_etnia_disponiveis) >= 4:
        # Quebra de página antes da seção de etnia (só se houver dados)
        st.markdown("""
        <div style="page-break-before: always; break-before: page;">
        </div>
        """, unsafe_allow_html=True)
        
        # Só exibir o header se houver dados
        st.markdown("""
        <div class="report-header" style="background: linear-gradient(135deg, #2ca02c, #1e7e34, #155724);">
            👥 PROFICIÊNCIA POR ETNIA
        </div>
        """, unsafe_allow_html=True)
        
        # Help para análise do gráfico
        with st.expander("ℹ️ Como analisar este gráfico", expanded=False):
            st.markdown("""
            **👥 Proficiência por Etnia - Informações Técnicas**
            
            **Construção do gráfico:**
            - **Tipo:** Gráfico de barras agrupadas (grouped bar chart)
            - **Eixo X:** Entidades (Estado, CREDE, Município, Escola)
            - **Eixo Y:** Taxa de participação (0% a 100%)
            - **Cores:** Baseadas na proficiência média de cada grupo étnico
            
            **O que representa:**
            - **Altura da barra:** Taxa de participação por grupo étnico
            - **Cor da barra:** Proficiência média do grupo (escala dinâmica)
            - **Grupos étnicos:** Preta, Branca, Parda, Amarela, Indígena
            
            **Como ler:**
            - **Altura:** Percentual de participação na avaliação
            - **Cor:** Nível de proficiência (🟠 Baixa, 🟡 Média, 🟢 Alta)
            - **Legenda:** Escala de proficiência dinâmica
            - **Hover:** Valores específicos de participação e proficiência
            
            **Dados disponíveis:**
            - **Taxa de Participação:** Percentual de alunos que participaram
            - **Proficiência Média:** Pontuação média do grupo
            - **Número de Alunos:** Quantidade de estudantes por grupo
            """)
        
        df_etnia = df_concat[colunas_etnia_disponiveis].copy()
        
        colunas_valores_etnia = ['VL_PRETA', 'VL_BRANCA', 'VL_PARDA', 'VL_AMARELA', 'VL_INDIGENA',
                                     'TX_PRETA','TX_BRANCA','TX_PARDA','TX_AMARELA','TX_INDIGENA',
                                     'NU_PRETA','NU_BRANCA','NU_PARDA','NU_AMARELA','NU_INDIGENA']
        df_etnia = converter_para_numerico(df_etnia, colunas_valores_etnia)
        
        colunas_etnia_valores = [col for col in colunas_valores_etnia if col in df_etnia.columns]
        if colunas_etnia_valores:
            df_etnia = df_etnia.dropna(subset=colunas_etnia_valores, how='all')
        
        if not df_etnia.empty:
            renomear = {
                'TP_ENTIDADE': 'Tipo de Entidade Código',
                'DC_TIPO_ENTIDADE': 'Tipo de Entidade',
                'NM_ENTIDADE': 'Entidade',
                'VL_FILTRO_DISCIPLINA': 'Componente Curricular',
                'VL_FILTRO_ETAPA': 'Etapa',
                'VL_PRETA': 'Proficiência Preta',
                'VL_BRANCA': 'Proficiência Branca',
                'VL_PARDA': 'Proficiência Parda',
                'VL_AMARELA': 'Proficiência Amarela',
                'VL_INDIGENA': 'Proficiência Indígena',
                'TX_PRETA': 'Taxa Preta',
                'TX_BRANCA': 'Taxa Branca',
                'TX_PARDA': 'Taxa Parda',
                'TX_AMARELA': 'Taxa Amarela',
                'TX_INDIGENA': 'Taxa Indígena',
                'NU_PRETA': 'Número Preta',
                'NU_BRANCA': 'Número Branca',
                'NU_PARDA': 'Número Parda',
                'NU_AMARELA': 'Número Amarela',
                'NU_INDIGENA': 'Número Indígena'
            }
            df_etnia = df_etnia.rename(columns={k: v for k, v in renomear.items() if k in df_etnia.columns})
            
            st.info(f"📊 {len(df_etnia)} registros com dados de proficiência por etnia")
            
            csv_etnia = df_etnia.to_csv(index=False).encode('utf-8')
            
            # Criar gráfico de barras para taxa e proficiência por Etnia
            st.subheader("Gráfico de Taxa e Proficiência por Etnia - Por Tipo de Entidade")
            
            # Verificar quais colunas estão disponíveis
            colunas_taxa_etnia = [col for col in ['Taxa Preta', 'Taxa Branca', 
                                                'Taxa Parda', 'Taxa Amarela', 
                                                'Taxa Indígena'] if col in df_etnia.columns]
            
            colunas_prof_etnia = [col for col in ['Proficiência Preta', 'Proficiência Branca', 
                                                'Proficiência Parda', 'Proficiência Amarela', 
                                                'Proficiência Indígena'] if col in df_etnia.columns]
            
            if colunas_taxa_etnia and colunas_prof_etnia and ('Tipo de Entidade Código' in df_etnia.columns or 'Tipo de Entidade' in df_etnia.columns):
                # Preparar dados para o gráfico
                df_plot = df_etnia.copy()
                
                # Mapear tipos de entidade para nomes amigáveis
                mapa_tipos = {
                    '01': 'Ceará',
                    '02': 'CREDE',
                    '11': 'Município',
                    '03': 'Escola',
                    'Estado': 'Ceará',
                    'Regional': 'CREDE',
                    'Município': 'Município',
                    'Escola': 'Escola'
                }
                
                # Criar coluna com tipo de entidade amigável
                if 'Tipo de Entidade Código' in df_plot.columns:
                    df_plot['Tipo Simplificado'] = df_plot['Tipo de Entidade Código'].map(mapa_tipos)
                    if df_plot['Tipo Simplificado'].isna().any() and 'Tipo de Entidade' in df_plot.columns:
                        df_plot['Tipo Simplificado'] = df_plot['Tipo Simplificado'].fillna(
                            df_plot['Tipo de Entidade'].map(mapa_tipos)
                        )
                elif 'Tipo de Entidade' in df_plot.columns:
                    df_plot['Tipo Simplificado'] = df_plot['Tipo de Entidade'].map(mapa_tipos)
                
                # Incluir colunas de número
                colunas_numero_etnia = [col for col in ['Número Preta', 'Número Branca', 
                                                    'Número Parda', 'Número Amarela', 
                                                    'Número Indígena'] if col in df_etnia.columns]
                
                # Agrupar por tipo de entidade e calcular a média
                todas_colunas = colunas_taxa_etnia + colunas_prof_etnia + colunas_numero_etnia
                df_plot = df_plot.groupby('Tipo Simplificado')[todas_colunas].mean().reset_index()
                
                # Criar lista de dados para o gráfico
                dados_grafico = []
                
                # Categorias de etnia
                categorias = {
                    'Preta': {'taxa': 'Taxa Preta', 'prof': 'Proficiência Preta', 'numero': 'Número Preta', 'cor_base': COR_PRIMARIA},
                    'Branca': {'taxa': 'Taxa Branca', 'prof': 'Proficiência Branca', 'numero': 'Número Branca', 'cor_base': COR_SECUNDARIA},
                    'Parda': {'taxa': 'Taxa Parda', 'prof': 'Proficiência Parda', 'numero': 'Número Parda', 'cor_base': COR_SUCESSO},
                    'Amarela': {'taxa': 'Taxa Amarela', 'prof': 'Proficiência Amarela', 'numero': 'Número Amarela', 'cor_base': COR_DANGER},
                    'Indígena': {'taxa': 'Taxa Indígena', 'prof': 'Proficiência Indígena', 'numero': 'Número Indígena', 'cor_base': COR_LIGHT}
                }
                
                # Função para calcular cor baseada na proficiência (laranja -> verde)
                def calcular_cor_intensidade(cor_base_hex, proficiencia, prof_min, prof_max):
                    """Calcula a cor variando de laranja (baixa proficiência) a verde (alta proficiência)"""
                    # Cores: Laranja para proficiência baixa, Verde para proficiência alta
                    # Laranja: #FF6B35 (255, 107, 53)
                    # Amarelo intermediário: #FFB830 (255, 184, 48)
                    # Verde claro: #87C147 (135, 193, 71)
                    # Verde escuro: #2E7D32 (46, 125, 50)
                    
                    # Normalizar proficiência entre 0 e 1
                    if prof_max > prof_min:
                        normalizado = (proficiencia - prof_min) / (prof_max - prof_min)
                    else:
                        normalizado = 0.5
                    
                    # Interpolar cores de acordo com a proficiência
                    if normalizado < 0.33:  # Laranja a Amarelo
                        # Escalar de 0-0.33 para 0-1
                        t = normalizado / 0.33
                        r = int(255)  # Mantém vermelho alto
                        g = int(107 + (184 - 107) * t)  # De 107 a 184
                        b = int(53 + (48 - 53) * t)  # De 53 a 48
                    elif normalizado < 0.67:  # Amarelo a Verde claro
                        # Escalar de 0.33-0.67 para 0-1
                        t = (normalizado - 0.33) / 0.34
                        r = int(255 - (255 - 135) * t)  # De 255 a 135
                        g = int(184 + (193 - 184) * t)  # De 184 a 193
                        b = int(48 + (71 - 48) * t)  # De 48 a 71
                    else:  # Verde claro a Verde escuro
                        # Escalar de 0.67-1.0 para 0-1
                        t = (normalizado - 0.67) / 0.33
                        r = int(135 - (135 - 46) * t)  # De 135 a 46
                        g = int(193 - (193 - 125) * t)  # De 193 a 125
                        b = int(71 - (71 - 50) * t)  # De 71 a 50
                    
                    return f'rgb({r},{g},{b})'
                
                # Calcular proficiência mínima e máxima para normalização
                todas_proficiencias = []
                for cat_info in categorias.values():
                    if cat_info['prof'] in df_plot.columns:
                        todas_proficiencias.extend(df_plot[cat_info['prof']].dropna().tolist())
                
                prof_min = min(todas_proficiencias) if todas_proficiencias else 0
                prof_max = max(todas_proficiencias) if todas_proficiencias else 100
                
                # Preparar dados para cada tipo de entidade e categoria
                for tipo_entidade in df_plot['Tipo Simplificado']:
                    for cat_nome, cat_info in categorias.items():
                        if cat_info['taxa'] in df_plot.columns and cat_info['prof'] in df_plot.columns and cat_info['numero'] in df_plot.columns:
                            dados_tipo = df_plot[df_plot['Tipo Simplificado'] == tipo_entidade]
                            if not dados_tipo.empty:
                                taxa = dados_tipo[cat_info['taxa']].values[0]
                                proficiencia = dados_tipo[cat_info['prof']].values[0]
                                numero = dados_tipo[cat_info['numero']].values[0]
                                
                                if pd.notna(taxa) and pd.notna(proficiencia) and pd.notna(numero):
                                    cor = calcular_cor_intensidade(cat_info['cor_base'], proficiencia, prof_min, prof_max)
                                    
                                    dados_grafico.append({
                                        'Tipo de Entidade': tipo_entidade,
                                        'Etnia': cat_nome,
                                        'Taxa': taxa,
                                        'Proficiência': proficiencia,
                                        'Numero': numero,
                                        'Cor': cor,
                                        'Label': f"{cat_nome}<br>{taxa:.1f}%".replace('.', ',')
                                    })
                
                # Criar DataFrame dos dados
                df_grafico = pd.DataFrame(dados_grafico)
                
                if not df_grafico.empty:
                    # Criar gráfico
                    fig_etnia = go.Figure()
                    
                    # Definir ordem dos tipos de entidade
                    ordem_tipos = ['Ceará', 'CREDE', 'Município', 'Escola']
                    
                    # Adicionar barras para cada etnia
                    categorias_etnias = df_grafico['Etnia'].unique()
                    
                    for etnia in categorias_etnias:
                        df_etnia_cat = df_grafico[df_grafico['Etnia'] == etnia].copy()
                        
                        # Ordenar pelo tipo de entidade usando a ordem definida
                        df_etnia_cat['Ordem'] = df_etnia_cat['Tipo de Entidade'].map({t: i for i, t in enumerate(ordem_tipos)})
                        df_etnia_cat = df_etnia_cat.sort_values('Ordem')
                        
                        fig_etnia.add_trace(go.Bar(
                            name=etnia,
                            x=df_etnia_cat['Tipo de Entidade'],
                            y=df_etnia_cat['Taxa'],
                            marker=dict(
                                color=df_etnia_cat['Cor'].tolist(),
                                line=dict(color='rgba(0,0,0,0.3)', width=1)
                            ),
                            text=[f"{e}<br>{t:.1f}%<br>Prof: {p:.0f}<br>N: {n:.0f}".replace('.', ',') for e, t, p, n in zip(df_etnia_cat['Etnia'], df_etnia_cat['Taxa'], df_etnia_cat['Proficiência'], df_etnia_cat['Numero'])],
                            textposition='outside',
                            textfont=dict(size=12, family='Arial', color='black'),
                            textangle=-90,
                            hovertemplate='<b style="font-size:18px">Tipo: %{x}</b><br><span style="font-size:16px">Etnia: ' + etnia + '<br>Percentual de Alunos: %{y:.1f}%<br>Proficiência: %{customdata[0]:.1f}<br>Número de Alunos: %{customdata[1]:,}</span><extra></extra>'.replace('%{y:.1f}%', '%{y:.1f}%').replace('%{customdata[0]:.1f}', '%{customdata[0]:.1f}').replace('.', ','),
                            customdata=list(zip(df_etnia_cat['Proficiência'], df_etnia_cat['Numero'])),
                            showlegend=False
                        ))
                    
                    # Tipos disponíveis na ordem correta
                    tipos_disponiveis = [t for t in ordem_tipos if t in df_grafico['Tipo de Entidade'].unique()]
                    
                    # Configurar o layout do gráfico
                    fig_etnia.update_layout(
                        title=dict(
                            text=f'👥 Taxa (altura) e Proficiência (cor) por Etnia<br><sub style="font-size:14px;">🟠 Laranja = Proficiência Baixa | 🟡 Amarelo = Proficiência Média | 🟢 Verde = Proficiência Alta | Escala: {prof_min:.0f} - {prof_max:.0f}</sub>',
                            font=dict(size=18, family='Arial Black')
                        ),
                        xaxis_title=dict(
                            text='Tipo de Entidade',
                            font=dict(size=18)
                        ),
                        yaxis_title=dict(
                            text='Taxa (%)',
                            font=dict(size=18)
                        ),
                        font=dict(size=14),
                        height=450,
                        barmode='group',
                        bargap=0.2,
                        bargroupgap=0.15,
                        yaxis=dict(
                            range=[0, 110],
                            tickfont=dict(size=14)
                        ),
                        showlegend=False,
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=tipos_disponiveis,
                            tickfont=dict(size=14)
                        ),
                        hoverlabel=dict(
                            font_size=20,
                            font_family="Arial"
                        )
                    )
                    
                    # Exibir o gráfico
                    st.plotly_chart(fig_etnia, use_container_width=True)
                else:
                    st.info("Não foi possível gerar o gráfico com os dados disponíveis")
            else:
                st.info("Dados de taxa e proficiência por etnia insuficientes para gerar o gráfico ou coluna de tipo de entidade não disponível")
            
            st.download_button(
                "📥 Baixar Dados de Etnia",
                data=csv_etnia,
                file_name="proficiencia_etnia.csv",
                mime="text/csv",
                key="download_etnia"
            )
            
            # Análise com Groq
            with st.expander("🤖 Análise Inteligente - Proficiência por Etnia", expanded=False):
                if st.session_state.get('documentos_carregados', False) and st.session_state.get('ia_ativa', True):
                    st.error("⚠️ **Lembrete:** Esta análise é gerada por inteligência artificial e pode conter erros ou imprecisões. **Esta funcionalidade está em fase de testes.** Use sempre seu julgamento profissional para validar as informações.")
                    
                    # Criar chave única baseada nos filtros atuais
                    etapa_filtro = st.session_state.get('etapa_selecionada', 'Todas')
                    disciplina_filtro = st.session_state.get('disciplina_selecionada', 'Todas')
                    key_analise = f"analise_etnia_{etapa_filtro}_{disciplina_filtro}"
                    
                    if st.button("🔍 Analisar Dados com IA", key=key_analise):
                        with st.spinner("🤖 Analisando dados com IA..."):
                            analise = analisar_dataframe_com_groq(
                                df_etnia, 
                                "Proficiência por Etnia", 
                            "Análise das diferenças de proficiência entre grupos étnicos nas avaliações SPAECE",
                            st.session_state.agregado_consultado,
                            st.session_state.df_concatenado
                        )
                        st.markdown(analise)
                elif not st.session_state.get('documentos_carregados', False):
                    st.warning("⚠️ **Análise IA indisponível:** Carregue as bases de dados (DCRC e BNCC) para ativar as análises inteligentes.")
                else:
                    st.warning("⚠️ **Análise IA desativada:** Use o botão no painel lateral para ativar as análises inteligentes.")
        else:
            st.info("Sem dados válidos de proficiência por etnia após limpeza")
    else:
        st.info("Colunas de etnia não encontradas no conjunto de dados")
    
    
    # Quebra de página antes da seção de NSE (removida para evitar páginas vazias)
    # st.markdown("""
    # <div style="page-break-before: always; break-before: page;">
    # </div>
    # """, unsafe_allow_html=True)
    
    # ==================== DADOS CONTEXTUAIS - NSE ====================
    colunas_nse = ['TP_ENTIDADE', 'DC_TIPO_ENTIDADE', 'NM_ENTIDADE', 'VL_FILTRO_DISCIPLINA', 'VL_FILTRO_ETAPA', 'VL_NSE1', 
                   'VL_NSE2', 'VL_NSE3', 'VL_NSE4', 'TX_NSE1','TX_NSE2','TX_NSE3','TX_NSE4',
                   'NU_NSE1','NU_NSE2','NU_NSE3','NU_NSE4']
    colunas_nse_disponiveis = [col for col in colunas_nse if col in df_concat.columns]
    
    if len(colunas_nse_disponiveis) >= 4:
        # Quebra de página antes da seção de NSE (só se houver dados)
        st.markdown("""
        <div style="page-break-before: always; break-before: page;">
        </div>
        """, unsafe_allow_html=True)
        
        # Só exibir o header se houver dados
        st.markdown(f"""
        <div class="report-header" style="background: linear-gradient(135deg, {COR_SECUNDARIA}, #e67e22, #d35400);">
            💰 PROFICIÊNCIA POR NÍVEL SOCIOECONÔMICO (NSE)
        </div>
        """, unsafe_allow_html=True)
        
        # Help para análise do gráfico
        with st.expander("ℹ️ Como analisar este gráfico", expanded=False):
            st.markdown("""
            **💰 Proficiência por Nível Socioeconômico (NSE) - Informações Técnicas**
            
            **Construção do gráfico:**
            - **Tipo:** Gráfico de barras agrupadas (grouped bar chart)
            - **Eixo X:** Entidades (Estado, CREDE, Município, Escola)
            - **Eixo Y:** Taxa de participação (0% a 100%)
            - **Cores:** Baseadas na proficiência média de cada nível NSE
            
            **O que representa:**
            - **Altura da barra:** Taxa de participação por nível NSE
            - **Cor da barra:** Proficiência média do nível (escala dinâmica)
            - **Níveis NSE:** NSE 1 (mais baixo) a NSE 4 (mais alto)
            
            **Como ler:**
            - **Altura:** Percentual de participação na avaliação
            - **Cor:** Nível de proficiência (🟠 Baixa, 🟡 Média, 🟢 Alta)
            - **Legenda:** Escala de proficiência dinâmica
            - **Hover:** Valores específicos de participação e proficiência
            
            **Dados disponíveis:**
            - **Taxa de Participação:** Percentual de alunos que participaram
            - **Proficiência Média:** Pontuação média do nível NSE
            - **Número de Alunos:** Quantidade de estudantes por nível
            """)
        
        df_nse = df_concat[colunas_nse_disponiveis].copy()
        
        colunas_valores_nse = ['VL_NSE1', 'VL_NSE2', 'VL_NSE3', 'VL_NSE4', 
                              'NU_NSE1', 'NU_NSE2', 'NU_NSE3', 'NU_NSE4',
                              'TX_NSE1','TX_NSE2','TX_NSE3','TX_NSE4']
        df_nse = converter_para_numerico(df_nse, colunas_valores_nse)
        
        colunas_nse_valores = [col for col in colunas_valores_nse if col in df_nse.columns]
        if colunas_nse_valores:
            df_nse = df_nse.dropna(subset=colunas_nse_valores, how='all')
        
        if not df_nse.empty:
            renomear = {
                'TP_ENTIDADE': 'Tipo de Entidade Código',
                'DC_TIPO_ENTIDADE': 'Tipo de Entidade',
                'NM_ENTIDADE': 'Entidade',
                'VL_FILTRO_DISCIPLINA': 'Componente Curricular',
                'VL_FILTRO_ETAPA': 'Etapa',
                'VL_NSE1': 'Proficiência NSE 1 (Mais Baixo)',
                'VL_NSE2': 'Proficiência NSE 2',
                'VL_NSE3': 'Proficiência NSE 3',
                'VL_NSE4': 'Proficiência NSE 4 (Mais Alto)',
                'NU_NSE1': 'Número NSE 1',
                'NU_NSE2': 'Número NSE 2',
                'NU_NSE3': 'Número NSE 3',
                'NU_NSE4': 'Número NSE 4',
                'TX_NSE1': 'Taxa NSE 1',
                'TX_NSE2': 'Taxa NSE 2',
                'TX_NSE3': 'Taxa NSE 3',
                'TX_NSE4': 'Taxa NSE 4'
            }
            df_nse = df_nse.rename(columns={k: v for k, v in renomear.items() if k in df_nse.columns})
            
            st.info(f"📊 {len(df_nse)} registros com dados de proficiência por NSE")
            # st.dataframe(df_nse, use_container_width=True, height=400)
            
            # Criar gráfico para NSE
            st.subheader("Gráfico de Taxa e Proficiência por NSE - Por Tipo de Entidade")
            
            # Verificar quais colunas estão disponíveis
            colunas_taxa_nse = [col for col in ['Taxa NSE 1', 'Taxa NSE 2', 
                                                'Taxa NSE 3', 'Taxa NSE 4'] if col in df_nse.columns]
            
            colunas_prof_nse = [col for col in ['Proficiência NSE 1 (Mais Baixo)', 'Proficiência NSE 2', 
                                                'Proficiência NSE 3', 'Proficiência NSE 4 (Mais Alto)'] if col in df_nse.columns]
            
            if colunas_taxa_nse and colunas_prof_nse and ('Tipo de Entidade Código' in df_nse.columns or 'Tipo de Entidade' in df_nse.columns):
                # Preparar dados para o gráfico
                df_plot_nse = df_nse.copy()
                
                # Mapear tipos de entidade para nomes amigáveis
                mapa_tipos = {
                    '01': 'Ceará',
                    '02': 'CREDE',
                    '11': 'Município',
                    '03': 'Escola',
                    'Estado': 'Ceará',
                    'Regional': 'CREDE',
                    'Município': 'Município',
                    'Escola': 'Escola'
                }
                
                # Criar coluna com tipo de entidade amigável
                if 'Tipo de Entidade Código' in df_plot_nse.columns:
                    df_plot_nse['Tipo Simplificado'] = df_plot_nse['Tipo de Entidade Código'].map(mapa_tipos)
                    if df_plot_nse['Tipo Simplificado'].isna().any() and 'Tipo de Entidade' in df_plot_nse.columns:
                        df_plot_nse['Tipo Simplificado'] = df_plot_nse['Tipo Simplificado'].fillna(
                            df_plot_nse['Tipo de Entidade'].map(mapa_tipos)
                        )
                elif 'Tipo de Entidade' in df_plot_nse.columns:
                    df_plot_nse['Tipo Simplificado'] = df_plot_nse['Tipo de Entidade'].map(mapa_tipos)
                
                # Agrupar por tipo de entidade e calcular a média
                colunas_numero_nse = [col for col in ['Número NSE 1', 'Número NSE 2', 
                                                    'Número NSE 3', 'Número NSE 4'] if col in df_plot_nse.columns]
                todas_colunas_nse = colunas_taxa_nse + colunas_prof_nse + colunas_numero_nse
                df_plot_nse = df_plot_nse.groupby('Tipo Simplificado')[todas_colunas_nse].mean().reset_index()
                
                # Criar lista de dados para o gráfico
                dados_grafico_nse = []
                
                # Categorias de NSE
                categorias_nse = {
                     'NSE 1 (Mais Baixo)': {'taxa': 'Taxa NSE 1', 'prof': 'Proficiência NSE 1 (Mais Baixo)', 'numero': 'Número NSE 1', 'cor_base': COR_DANGER},
                     'NSE 2': {'taxa': 'Taxa NSE 2', 'prof': 'Proficiência NSE 2', 'numero': 'Número NSE 2', 'cor_base': COR_SECUNDARIA},
                     'NSE 3': {'taxa': 'Taxa NSE 3', 'prof': 'Proficiência NSE 3', 'numero': 'Número NSE 3', 'cor_base': COR_PRIMARIA},
                     'NSE 4 (Mais Alto)': {'taxa': 'Taxa NSE 4', 'prof': 'Proficiência NSE 4 (Mais Alto)', 'numero': 'Número NSE 4', 'cor_base': COR_SUCESSO}
                }
                
                # Função para calcular cor baseada na proficiência
                def calcular_cor_intensidade_nse(cor_base_hex, proficiencia, prof_min, prof_max):
                    if prof_max > prof_min:
                        normalizado = (proficiencia - prof_min) / (prof_max - prof_min)
                    else:
                        normalizado = 0.5
                    
                    if normalizado < 0.33:
                        t = normalizado / 0.33
                        r = int(255)
                        g = int(107 + (184 - 107) * t)
                        b = int(53 + (48 - 53) * t)
                    elif normalizado < 0.67:
                        t = (normalizado - 0.33) / 0.34
                        r = int(255 - (255 - 135) * t)
                        g = int(184 + (193 - 184) * t)
                        b = int(48 + (71 - 48) * t)
                    else:
                        t = (normalizado - 0.67) / 0.33
                        r = int(135 - (135 - 46) * t)
                        g = int(193 - (193 - 125) * t)
                        b = int(71 - (71 - 50) * t)
                    
                    return f'rgb({r},{g},{b})'
                
                # Calcular proficiência mínima e máxima
                todas_proficiencias_nse = []
                for cat_info in categorias_nse.values():
                    if cat_info['prof'] in df_plot_nse.columns:
                        todas_proficiencias_nse.extend(df_plot_nse[cat_info['prof']].dropna().tolist())
                
                prof_min_nse = min(todas_proficiencias_nse) if todas_proficiencias_nse else 0
                prof_max_nse = max(todas_proficiencias_nse) if todas_proficiencias_nse else 100
                
                # Preparar dados para cada tipo de entidade e categoria
                for tipo_entidade in df_plot_nse['Tipo Simplificado']:
                    for cat_nome, cat_info in categorias_nse.items():
                        if cat_info['taxa'] in df_plot_nse.columns and cat_info['prof'] in df_plot_nse.columns and cat_info['numero'] in df_plot_nse.columns:
                            dados_tipo = df_plot_nse[df_plot_nse['Tipo Simplificado'] == tipo_entidade]
                            if not dados_tipo.empty:
                                taxa = dados_tipo[cat_info['taxa']].values[0]
                                proficiencia = dados_tipo[cat_info['prof']].values[0]
                                numero = dados_tipo[cat_info['numero']].values[0]
                                
                                if pd.notna(taxa) and pd.notna(proficiencia) and pd.notna(numero):
                                    cor = calcular_cor_intensidade_nse(cat_info['cor_base'], proficiencia, prof_min_nse, prof_max_nse)
                                    
                                    dados_grafico_nse.append({
                                        'Tipo de Entidade': tipo_entidade,
                                        'NSE': cat_nome,
                                        'Taxa': taxa,
                                        'Proficiência': proficiencia,
                                        'Numero': numero,
                                        'Cor': cor,
                                        'Label': f"{cat_nome}<br>{taxa:.1f}%".replace('.', ',')
                                    })
                
                # Criar DataFrame dos dados
                df_grafico_nse = pd.DataFrame(dados_grafico_nse)
                
                if not df_grafico_nse.empty:
                    # Criar gráfico
                    fig_nse = go.Figure()
                    
                    # Definir ordem dos tipos de entidade
                    ordem_tipos = ['Ceará', 'CREDE', 'Município', 'Escola']
                    
                    # Adicionar barras para cada NSE
                    categorias_nse_grafico = df_grafico_nse['NSE'].unique()
                    
                    for nse in categorias_nse_grafico:
                        df_nse_cat = df_grafico_nse[df_grafico_nse['NSE'] == nse].copy()
                        
                        # Ordenar pelo tipo de entidade
                        df_nse_cat['Ordem'] = df_nse_cat['Tipo de Entidade'].map({t: i for i, t in enumerate(ordem_tipos)})
                        df_nse_cat = df_nse_cat.sort_values('Ordem')
                        
                        fig_nse.add_trace(go.Bar(
                            name=nse,
                            x=df_nse_cat['Tipo de Entidade'],
                            y=df_nse_cat['Taxa'],
                            marker=dict(
                                color=df_nse_cat['Cor'].tolist(),
                                line=dict(color='rgba(0,0,0,0.3)', width=1)
                            ),
                             text=[f"{n}<br>{t:.1f}%<br>Prof: {p:.0f}<br>N: {num:.0f}".replace('.', ',') for n, t, p, num in zip(df_nse_cat['NSE'], df_nse_cat['Taxa'], df_nse_cat['Proficiência'], df_nse_cat['Numero'])],
                             textposition='outside',
                             textfont=dict(size=12, family='Arial', color='black'),
                             textangle=-90,
                            hovertemplate='<b style="font-size:18px">Tipo: %{x}</b><br><span style="font-size:16px">NSE: ' + nse + '<br>Taxa: %{y:.1f}%<br>Proficiência: %{customdata:.1f}</span><extra></extra>'.replace('%{y:.1f}%', '%{y:.1f}%').replace('%{customdata:.1f}', '%{customdata:.1f}').replace('.', ','),
                            customdata=df_nse_cat['Proficiência'],
                            showlegend=False
                        ))
                    
                    # Tipos disponíveis na ordem correta
                    tipos_disponiveis_nse = [t for t in ordem_tipos if t in df_grafico_nse['Tipo de Entidade'].unique()]
                    
                    # Configurar o layout do gráfico
                    fig_nse.update_layout(
                        title=dict(
                            text=f'📊 Taxa (altura) e Proficiência (cor) por NSE<br><sub style="font-size:14px;">🟠 Laranja = Proficiência Baixa | 🟡 Amarelo = Proficiência Média | 🟢 Verde = Proficiência Alta | Escala: {prof_min_nse:.0f} - {prof_max_nse:.0f}</sub>',
                            font=dict(size=18, family='Arial Black')
                        ),
                        xaxis_title=dict(
                            text='Tipo de Entidade',
                            font=dict(size=18)
                        ),
                        yaxis_title=dict(
                            text='Taxa (%)',
                            font=dict(size=18)
                        ),
                        font=dict(size=14),
                        height=450,
                        barmode='group',
                        bargap=0.2,
                        bargroupgap=0.15,
                        yaxis=dict(
                            range=[0, 110],
                            tickfont=dict(size=14)
                        ),
                        showlegend=False,
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=tipos_disponiveis_nse,
                            tickfont=dict(size=14)
                        ),
                        hoverlabel=dict(
                            font_size=20,
                            font_family="Arial"
                        )
                    )
                    
                    # Exibir o gráfico
                    st.plotly_chart(fig_nse, use_container_width=True)
                else:
                    st.info("Não foi possível gerar o gráfico com os dados disponíveis")
            else:
                st.info("Dados de taxa e proficiência por NSE insuficientes para gerar o gráfico ou coluna de tipo de entidade não disponível")
            
            csv_nse = df_nse.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Baixar Dados de NSE",
                data=csv_nse,
                file_name="proficiencia_nse.csv",
                mime="text/csv",
                key="download_nse"
            )
            
            # Análise com Groq
            with st.expander("🤖 Análise Inteligente - Proficiência por NSE", expanded=False):
                if st.session_state.get('documentos_carregados', False) and st.session_state.get('ia_ativa', True):
                    st.error("⚠️ **Lembrete:** Esta análise é gerada por inteligência artificial e pode conter erros ou imprecisões. **Esta funcionalidade está em fase de testes.** Use sempre seu julgamento profissional para validar as informações.")
                    
                    # Criar chave única baseada nos filtros atuais
                    etapa_filtro = st.session_state.get('etapa_selecionada', 'Todas')
                    disciplina_filtro = st.session_state.get('disciplina_selecionada', 'Todas')
                    key_analise = f"analise_nse_{etapa_filtro}_{disciplina_filtro}"
                    
                    if st.button("🔍 Analisar Dados com IA", key=key_analise):
                        with st.spinner("🤖 Analisando dados com IA..."):
                            analise = analisar_dataframe_com_groq(
                                df_nse, 
                                "Proficiência por NSE", 
                            "Análise das diferenças de proficiência entre níveis socioeconômicos nas avaliações SPAECE",
                            st.session_state.agregado_consultado,
                            st.session_state.df_concatenado
                        )
                        st.markdown(analise)
                elif not st.session_state.get('documentos_carregados', False):
                    st.warning("⚠️ **Análise IA indisponível:** Carregue as bases de dados (DCRC e BNCC) para ativar as análises inteligentes.")
                else:
                    st.warning("⚠️ **Análise IA desativada:** Use o botão no painel lateral para ativar as análises inteligentes.")
    
    # ==================== DADOS CONTEXTUAIS - SEXO ====================
    colunas_sexo = ['TP_ENTIDADE', 'DC_TIPO_ENTIDADE', 'NM_ENTIDADE', 'VL_FILTRO_DISCIPLINA', 'VL_FILTRO_ETAPA', 'VL_FEMININO', 
                   'VL_MASCULINO', 'TX_FEMININO','TX_MASCULINO','NU_FEMININO','NU_MASCULINO']
    colunas_sexo_disponiveis = [col for col in colunas_sexo if col in df_concat.columns]
    
    if len(colunas_sexo_disponiveis) >= 4:
        # Quebra de página antes da seção de sexo (só se houver dados)
        st.markdown("""
        <div style="page-break-before: always; break-before: page;">
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="report-header" style="background: linear-gradient(135deg, #2ca02c, #1e7e34, #155724);">
            👫 PROFICIÊNCIA POR SEXO
        </div>
        """, unsafe_allow_html=True)
        
        # Help para análise do gráfico
        with st.expander("ℹ️ Como analisar este gráfico", expanded=False):
            st.markdown("""
            **👫 Proficiência por Sexo - Informações Técnicas**
            
            **Construção do gráfico:**
            - **Tipo:** Gráfico de barras agrupadas (grouped bar chart)
            - **Eixo X:** Entidades (Estado, CREDE, Município, Escola)
            - **Eixo Y:** Taxa de participação (0% a 100%)
            - **Cores:** Baseadas na proficiência média de cada sexo
            
            **O que representa:**
            - **Altura da barra:** Taxa de participação por sexo
            - **Cor da barra:** Proficiência média do sexo (escala dinâmica)
            - **Grupos:** Feminino e Masculino
            
            **Como ler:**
            - **Altura:** Percentual de participação na avaliação
            - **Cor:** Nível de proficiência (🟠 Baixa, 🟡 Média, 🟢 Alta)
            - **Legenda:** Escala de proficiência dinâmica
            - **Hover:** Valores específicos de participação e proficiência
            
            **Dados disponíveis:**
            - **Taxa de Participação:** Percentual de alunos que participaram
            - **Proficiência Média:** Pontuação média por sexo
            - **Número de Alunos:** Quantidade de estudantes por sexo
            """)
        
        df_sexo = df_concat[colunas_sexo_disponiveis].copy()
        
        colunas_valores_sexo = ['VL_FEMININO', 'VL_MASCULINO', 'NU_FEMININO', 
                               'NU_MASCULINO', 'TX_FEMININO', 'TX_MASCULINO']
        df_sexo = converter_para_numerico(df_sexo, colunas_valores_sexo)
        
        colunas_sexo_valores = [col for col in colunas_valores_sexo if col in df_sexo.columns]
        if colunas_sexo_valores:
            df_sexo = df_sexo.dropna(subset=colunas_sexo_valores, how='all')
        
        if not df_sexo.empty:
            renomear = {
                'TP_ENTIDADE': 'Tipo de Entidade Código',
                'DC_TIPO_ENTIDADE': 'Tipo de Entidade',
                'NM_ENTIDADE': 'Entidade',
                'VL_FILTRO_DISCIPLINA': 'Componente Curricular',
                'VL_FILTRO_ETAPA': 'Etapa',
                'VL_FEMININO': 'Proficiência Feminino',
                'VL_MASCULINO': 'Proficiência Masculino',
                'NU_FEMININO': 'Número Feminino',
                'NU_MASCULINO': 'Número Masculino',
                'TX_FEMININO': 'Taxa Feminino',
                'TX_MASCULINO': 'Taxa Masculino'
            }
            df_sexo = df_sexo.rename(columns={k: v for k, v in renomear.items() if k in df_sexo.columns})
            
            st.info(f"📊 {len(df_sexo)} registros com dados de proficiência por sexo")
            # st.dataframe(df_sexo, use_container_width=True, height=400)
            
            # Criar gráfico para Sexo
            st.subheader("Gráfico de Taxa e Proficiência por Sexo - Por Tipo de Entidade")
            
            # Verificar quais colunas estão disponíveis
            colunas_taxa_sexo = [col for col in ['Taxa Feminino', 'Taxa Masculino'] if col in df_sexo.columns]
            
            colunas_prof_sexo = [col for col in ['Proficiência Feminino', 'Proficiência Masculino'] if col in df_sexo.columns]
            
            if colunas_taxa_sexo and colunas_prof_sexo and ('Tipo de Entidade Código' in df_sexo.columns or 'Tipo de Entidade' in df_sexo.columns):
                # Preparar dados para o gráfico
                df_plot_sexo = df_sexo.copy()
                
                # Mapear tipos de entidade para nomes amigáveis
                mapa_tipos = {
                    '01': 'Ceará',
                    '02': 'CREDE',
                    '11': 'Município',
                    '03': 'Escola',
                    'Estado': 'Ceará',
                    'Regional': 'CREDE',
                    'Município': 'Município',
                    'Escola': 'Escola'
                }
                
                # Criar coluna com tipo de entidade amigável
                if 'Tipo de Entidade Código' in df_plot_sexo.columns:
                    df_plot_sexo['Tipo Simplificado'] = df_plot_sexo['Tipo de Entidade Código'].map(mapa_tipos)
                    if df_plot_sexo['Tipo Simplificado'].isna().any() and 'Tipo de Entidade' in df_plot_sexo.columns:
                        df_plot_sexo['Tipo Simplificado'] = df_plot_sexo['Tipo Simplificado'].fillna(
                            df_plot_sexo['Tipo de Entidade'].map(mapa_tipos)
                        )
                elif 'Tipo de Entidade' in df_plot_sexo.columns:
                    df_plot_sexo['Tipo Simplificado'] = df_plot_sexo['Tipo de Entidade'].map(mapa_tipos)
                
                # Agrupar por tipo de entidade e calcular a média
                colunas_numero_sexo = [col for col in ['Número Feminino', 'Número Masculino'] if col in df_plot_sexo.columns]
                todas_colunas_sexo = colunas_taxa_sexo + colunas_prof_sexo + colunas_numero_sexo
                df_plot_sexo = df_plot_sexo.groupby('Tipo Simplificado')[todas_colunas_sexo].mean().reset_index()
                
                # Criar lista de dados para o gráfico
                dados_grafico_sexo = []
                
                # Categorias de Sexo
                categorias_sexo = {
                    'Feminino': {'taxa': 'Taxa Feminino', 'prof': 'Proficiência Feminino', 'numero': 'Número Feminino', 'cor_base': COR_SECUNDARIA},
                    'Masculino': {'taxa': 'Taxa Masculino', 'prof': 'Proficiência Masculino', 'numero': 'Número Masculino', 'cor_base': COR_PRIMARIA}
                }
                
                # Função para calcular cor baseada na proficiência
                def calcular_cor_intensidade_sexo(cor_base_hex, proficiencia, prof_min, prof_max):
                    if prof_max > prof_min:
                        normalizado = (proficiencia - prof_min) / (prof_max - prof_min)
                    else:
                        normalizado = 0.5
                    
                    if normalizado < 0.33:
                        t = normalizado / 0.33
                        r = int(255)
                        g = int(107 + (184 - 107) * t)
                        b = int(53 + (48 - 53) * t)
                    elif normalizado < 0.67:
                        t = (normalizado - 0.33) / 0.34
                        r = int(255 - (255 - 135) * t)
                        g = int(184 + (193 - 184) * t)
                        b = int(48 + (71 - 48) * t)
                    else:
                        t = (normalizado - 0.67) / 0.33
                        r = int(135 - (135 - 46) * t)
                        g = int(193 - (193 - 125) * t)
                        b = int(71 - (71 - 50) * t)
                    
                    return f'rgb({r},{g},{b})'
                
                # Calcular proficiência mínima e máxima
                todas_proficiencias_sexo = []
                for cat_info in categorias_sexo.values():
                    if cat_info['prof'] in df_plot_sexo.columns:
                        todas_proficiencias_sexo.extend(df_plot_sexo[cat_info['prof']].dropna().tolist())
                
                prof_min_sexo = min(todas_proficiencias_sexo) if todas_proficiencias_sexo else 0
                prof_max_sexo = max(todas_proficiencias_sexo) if todas_proficiencias_sexo else 100
                
                # Preparar dados para cada tipo de entidade e categoria
                for tipo_entidade in df_plot_sexo['Tipo Simplificado']:
                    for cat_nome, cat_info in categorias_sexo.items():
                        if cat_info['taxa'] in df_plot_sexo.columns and cat_info['prof'] in df_plot_sexo.columns and cat_info['numero'] in df_plot_sexo.columns:
                            dados_tipo = df_plot_sexo[df_plot_sexo['Tipo Simplificado'] == tipo_entidade]
                            if not dados_tipo.empty:
                                taxa = dados_tipo[cat_info['taxa']].values[0]
                                proficiencia = dados_tipo[cat_info['prof']].values[0]
                                numero = dados_tipo[cat_info['numero']].values[0]
                                
                                if pd.notna(taxa) and pd.notna(proficiencia) and pd.notna(numero):
                                    cor = calcular_cor_intensidade_sexo(cat_info['cor_base'], proficiencia, prof_min_sexo, prof_max_sexo)
                                    
                                    dados_grafico_sexo.append({
                                        'Tipo de Entidade': tipo_entidade,
                                        'Sexo': cat_nome,
                                        'Taxa': taxa,
                                        'Proficiência': proficiencia,
                                        'Numero': numero,
                                        'Cor': cor,
                                        'Label': f"{cat_nome}<br>{taxa:.1f}%".replace('.', ',')
                                    })
                
                # Criar DataFrame dos dados
                df_grafico_sexo = pd.DataFrame(dados_grafico_sexo)
                
                if not df_grafico_sexo.empty:
                    # Criar gráfico
                    fig_sexo = go.Figure()
                    
                    # Definir ordem dos tipos de entidade
                    ordem_tipos = ['Ceará', 'CREDE', 'Município', 'Escola']
                    
                    # Adicionar barras para cada Sexo
                    categorias_sexo_grafico = df_grafico_sexo['Sexo'].unique()
                    
                    for sexo in categorias_sexo_grafico:
                        df_sexo_cat = df_grafico_sexo[df_grafico_sexo['Sexo'] == sexo].copy()
                        
                        # Ordenar pelo tipo de entidade
                        df_sexo_cat['Ordem'] = df_sexo_cat['Tipo de Entidade'].map({t: i for i, t in enumerate(ordem_tipos)})
                        df_sexo_cat = df_sexo_cat.sort_values('Ordem')
                        
                        fig_sexo.add_trace(go.Bar(
                            name=sexo,
                            x=df_sexo_cat['Tipo de Entidade'],
                            y=df_sexo_cat['Taxa'],
                            marker=dict(
                                color=df_sexo_cat['Cor'].tolist(),
                                line=dict(color='rgba(0,0,0,0.3)', width=1)
                            ),
                            text=[f"{s}<br>{t:.1f}%<br>Prof: {p:.0f}<br>N: {num:.0f}".replace('.', ',') for s, t, p, num in zip(df_sexo_cat['Sexo'], df_sexo_cat['Taxa'], df_sexo_cat['Proficiência'], df_sexo_cat['Numero'])],
                            textposition='outside',
                            textfont=dict(size=12, family='Arial', color='black'),
                            textangle=-90,
                            hovertemplate='<b style="font-size:18px">Tipo: %{x}</b><br><span style="font-size:16px">Sexo: ' + sexo + '<br>Taxa: %{y:.1f}%<br>Proficiência: %{customdata:.1f}</span><extra></extra>'.replace('%{y:.1f}%', '%{y:.1f}%').replace('%{customdata:.1f}', '%{customdata:.1f}').replace('.', ','),
                            customdata=df_sexo_cat['Proficiência'],
                            showlegend=False
                        ))
                    
                    # Tipos disponíveis na ordem correta
                    tipos_disponiveis_sexo = [t for t in ordem_tipos if t in df_grafico_sexo['Tipo de Entidade'].unique()]
                    
                    # Configurar o layout do gráfico
                    fig_sexo.update_layout(
                        title=dict(
                            text=f'👫 Taxa (altura) e Proficiência (cor) por Sexo<br><sub style="font-size:14px;">🟠 Laranja = Proficiência Baixa | 🟡 Amarelo = Proficiência Média | 🟢 Verde = Proficiência Alta | Escala: {prof_min_sexo:.0f} - {prof_max_sexo:.0f}</sub>',
                            font=dict(size=18, family='Arial Black')
                        ),
                        xaxis_title=dict(
                            text='Tipo de Entidade',
                            font=dict(size=18)
                        ),
                        yaxis_title=dict(
                            text='Taxa (%)',
                            font=dict(size=18)
                        ),
                        font=dict(size=14),
                        height=450,
                        barmode='group',
                        bargap=0.2,
                        bargroupgap=0.15,
                        yaxis=dict(
                            range=[0, 110],
                            tickfont=dict(size=14)
                        ),
                        showlegend=False,
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=tipos_disponiveis_sexo,
                            tickfont=dict(size=14)
                        ),
                        hoverlabel=dict(
                            font_size=20,
                            font_family="Arial"
                        )
                    )
                    
                    # Exibir o gráfico
                    st.plotly_chart(fig_sexo, use_container_width=True)
                else:
                    st.info("Não foi possível gerar o gráfico com os dados disponíveis")
            else:
                st.info("Dados de taxa e proficiência por sexo insuficientes para gerar o gráfico ou coluna de tipo de entidade não disponível")
            
            csv_sexo = df_sexo.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Baixar Dados de Sexo",
                data=csv_sexo,
                file_name="proficiencia_sexo.csv",
                mime="text/csv",
                key="download_sexo"
            )
            
            # Análise com Groq
            with st.expander("🤖 Análise Inteligente - Proficiência por Sexo", expanded=False):
                if st.session_state.get('documentos_carregados', False) and st.session_state.get('ia_ativa', True):
                    st.error("⚠️ **Lembrete:** Esta análise é gerada por inteligência artificial e pode conter erros ou imprecisões. **Esta funcionalidade está em fase de testes.** Use sempre seu julgamento profissional para validar as informações.")
                    
                    # Criar chave única baseada nos filtros atuais
                    etapa_filtro = st.session_state.get('etapa_selecionada', 'Todas')
                    disciplina_filtro = st.session_state.get('disciplina_selecionada', 'Todas')
                    key_analise = f"analise_sexo_{etapa_filtro}_{disciplina_filtro}"
                    
                    if st.button("🔍 Analisar Dados com IA", key=key_analise):
                        with st.spinner("🤖 Analisando dados com IA..."):
                            analise = analisar_dataframe_com_groq(
                                df_sexo, 
                                "Proficiência por Sexo", 
                            "Análise das diferenças de proficiência entre gêneros nas avaliações SPAECE",
                            st.session_state.agregado_consultado,
                            st.session_state.df_concatenado
                        )
                        st.markdown(analise)
                elif not st.session_state.get('documentos_carregados', False):
                    st.warning("⚠️ **Análise IA indisponível:** Carregue as bases de dados (DCRC e BNCC) para ativar as análises inteligentes.")
                else:
                    st.warning("⚠️ **Análise IA desativada:** Use o botão no painel lateral para ativar as análises inteligentes.")
        else:
            st.info("Sem dados válidos de proficiência por sexo após limpeza")
    else:
        st.info("Colunas de sexo não encontradas no conjunto de dados")
    
    # ==================== RESUMO EXECUTIVO ====================
    st.markdown("""
    <div class="report-header" style="background: linear-gradient(135deg, #2ca02c, #1e7e34, #155724);">
        📋 RESUMO EXECUTIVO
    </div>
    """, unsafe_allow_html=True)
    
    # Resumo executivo do relatório
    st.markdown("""
    <div class="report-card">
        <div class="report-card-header">
            📊 INFORMAÇÕES GERAIS DO RELATÓRIO
        </div>
        <div style="
            font-size: 0.95rem;
            line-height: 1.6;
            color: #374151;
        ">
            <p><strong>Data de Geração:</strong> {}</p>
            <p><strong>Agregado Consultado:</strong> {}</p>
            <p><strong>Total de Registros:</strong> {}</p>
            <p><strong>Período de Dados:</strong> Sistema Permanente de Avaliação da Educação Básica do Ceará (SPAECE)</p>
            <p><strong>Escopo:</strong> Análise educacional com foco em proficiência, participação e desempenho dos estudantes</p>
        </div>
    </div>
    """.format(
        pd.Timestamp.now().strftime("%d/%m/%Y às %H:%M"),
        st.session_state.agregado_consultado if st.session_state.agregado_consultado else "N/A",
        f"{len(st.session_state.df_concatenado):,}".replace(',', '.') if st.session_state.df_concatenado is not None else 0
    ), unsafe_allow_html=True)
    
    # Instruções de impressão
    st.markdown("""
    <div class="report-card">
        <div class="report-card-header" style="border-bottom-color: #f59c00;">
            🖨️ INSTRUÇÕES PARA IMPRESSÃO
        </div>
        <div style="
            font-size: 0.9rem;
            line-height: 1.6;
            color: #374151;
        ">
            <p><strong>Para salvar como PDF:</strong></p>
            <ul>
                <li>Use Ctrl+P (Windows/Linux) ou Cmd+P (Mac)</li>
                <li>Selecione "Salvar como PDF" como destino</li>
                <li><strong>Configurações recomendadas:</strong></li>
                <li style="margin-left: 1rem;">• Orientação: Paisagem (Landscape)</li>
                <li style="margin-left: 1rem;">• Margens: Mínimas (0.5in)</li>
                <li style="margin-left: 1rem;">• Escala: 100%</li>
                <li style="margin-left: 1rem;">• Gráficos de fundo: Ativado</li>
                <li style="margin-left: 1rem;">• Opções: Marcar "Mais configurações" e ativar "Gráficos de fundo"</li>
            </ul>

       </div>
    </div>
    """, unsafe_allow_html=True)
    
    

    st.markdown("<div style='text-align: center;'>Relatório SPAECE - CREDE 1 - Maracanaú</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'>Equipe Cecom 1</div>", unsafe_allow_html=True)
