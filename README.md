# SPAECE Results Dashboard

Dashboard interativo para análise dos resultados do SPAECE (Sistema Permanente de Avaliação da Educação Básica do Ceará).

## 📊 Funcionalidades

- **Análise Dinâmica**: Visualização de dados educacionais em tempo real
- **Filtros Inteligentes**: Seleção por etapa de ensino, disciplina e rede
- **IA Integrada**: Análise automática dos dados usando Groq API
- **Visualizações Avançadas**: Gráficos de proficiência, padrões de desempenho e distribuição
- **Autenticação**: Sistema de login para diferentes entidades (Estado, CREDE, Município, Escola)
- **Exportação**: Download de dados em CSV

## 🚀 Tecnologias Utilizadas

- **Streamlit**: Framework para aplicações web em Python
- **Pandas**: Manipulação e análise de dados
- **Plotly**: Visualizações interativas
- **Groq API**: Análise de dados com IA
- **Python 3.11+**: Linguagem principal

## 📋 Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/frfranklinrios/SPAECE.git
cd SPAECE
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as credenciais da API no arquivo `.streamlit/secrets.toml`

4. Execute a aplicação:
```bash
streamlit run streamlit_app.py
```

## 📁 Estrutura do Projeto

```
SPAECE/
├── streamlit_app.py          # Aplicação principal
├── config_api.py             # Configurações da API
├── gerador_indicadores.py    # Geração de indicadores
├── requirements.txt          # Dependências Python
├── .streamlit/               # Configurações do Streamlit
│   └── secrets.toml          # Credenciais da API
├── bncc.md                   # Documentação BNCC
├── dcrc.md                   # Documentação DCRC
└── README.md                 # Este arquivo
```

## 🎯 Como Usar

1. **Login**: Faça login com o código da entidade e senha
2. **Seleção**: Escolha a rede de ensino (Estadual/Municipal)
3. **Filtros**: Use os filtros para refinar a análise
4. **Análise**: Visualize os gráficos e métricas
5. **IA**: Use o botão de análise com IA para insights automáticos
6. **Export**: Baixe os dados em CSV quando necessário

## 📊 Escalas de Avaliação

- **Escala 500**: 2º ano do Ensino Fundamental
- **Escala 1000**: 5º ano, 9º ano e 3ª série do Ensino Médio

## 🔐 Autenticação

O sistema suporta diferentes níveis de acesso:
- **Estado**: Acesso completo a todos os dados
- **CREDE**: Dados da Coordenadoria Regional
- **Município**: Dados municipais específicos
- **Escola**: Dados da escola específica

## 📈 Indicadores Disponíveis

- Proficiência média por entidade
- Distribuição por padrão de desempenho
- Análise por nível socioeconômico
- Análise por gênero e raça/cor
- Taxa de participação
- Análise de habilidades específicas

## 🤖 Análise com IA

O sistema integra análise automática usando IA para:
- Identificar padrões nos dados
- Sugerir intervenções pedagógicas
- Analisar relações entre habilidades
- Gerar insights educacionais

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Franklin Rios**
- GitHub: [@frfranklinrios](https://github.com/frfranklinrios)

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Enviar pull requests

## 📞 Suporte

Para dúvidas ou suporte, entre em contato através do GitHub Issues.

---

**Desenvolvido para a Secretaria da Educação do Estado do Ceará (SEDUC-CE)**
