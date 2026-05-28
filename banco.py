import numpy as np

# ══════════════════════════════════════════════════════════
# PARÂMETROS DO SISTEMA

#   - LAMBDA: taxa média de chegada de clientes (processo de Poisson)
#   - MU:     taxa média de atendimento por caixa (distribuição exponencial)
#   - Os tempos entre eventos serão gerados como 1/taxa
# ══════════════════════════════════════════════════════════
LAMBDA    = 10   # chegadas por hora
MU        = 6    # atendimentos por hora por caixa
HORAS_DIA = 8    # banco abre 9h e fecha 17h → 8h de operação
DIAS      = 30   # total de dias úteis simulados


# ══════════════════════════════════════════════════════════
# FUNÇÃO: simular_dia
# Reproduz o funcionamento do banco em UM único dia.
# Recebe o número de caixas e devolve um dicionário com as
# métricas daquele dia.
# ══════════════════════════════════════════════════════════
def simular_dia(num_caixas: int) -> dict:

    # ── Geração das chegadas ───────────────────────────────
    # Em um processo de Poisson, o TEMPO ENTRE chegadas
    # consecutivas segue distribuição Exponencial(λ).
    # Geramos chegada por chegada, somando os intervalos,
    # até ultrapassar o horário de funcionamento.
    # Clientes que chegariam após o fechamento são ignorados.
    tempo, chegadas = 0.0, []
    while True:
        tempo += np.random.exponential(1 / LAMBDA)  # próximo intervalo
        if tempo >= HORAS_DIA:                       # chegaria fora do expediente?
            break
        chegadas.append(tempo)                       # registra o instante de chegada

    # Dia sem clientes: devolve métricas zeradas
    if not chegadas:
        return {"espera": 0.0, "atendidos": 0,
                "utilizacao": 0.0, "fila": 0.0}

    # ── Estado inicial dos caixas ──────────────────────────
    # Cada posição da lista representa um caixa.
    # O valor armazenado é o instante em que aquele caixa
    # ficará livre (começa em 0 → todos livres na abertura).
    caixas        = [0.0] * num_caixas

    # Acumula o tempo que cada caixa ficou efetivamente ocupado,
    # usado depois para calcular a taxa de utilização.
    tempo_ocupado = [0.0] * num_caixas

    esperas = []   # tempo de espera na fila de cada cliente

    # ── Processamento de cada cliente ─────────────────────
    for chegada in chegadas:

        # Escolhe o caixa que ficará livre primeiro
        # (estratégia ótima: cliente vai para a fila mais curta)
        idx = int(np.argmin(caixas))

        # O atendimento começa no maior entre:
        #   - o instante de chegada do cliente  (caixa já estava livre)
        #   - o instante em que o caixa libera  (cliente precisa esperar)
        inicio = max(chegada, caixas[idx])

        # Tempo que o cliente ficou esperando na fila
        espera = inicio - chegada

        # Tempo de serviço segue Exponencial(μ)
        t_servico = np.random.exponential(1 / MU)

        # Atualiza quando o caixa escolhido vai ficar livre novamente
        caixas[idx] = inicio + t_servico

        # Acumula o tempo produtivo deste caixa
        tempo_ocupado[idx] += t_servico

        esperas.append(espera)

    # ── Cálculo das métricas do dia ────────────────────────

    # Média dos tempos de espera de todos os clientes do dia
    espera_media = float(np.mean(esperas))

    # Quantidade de clientes que chegaram dentro do expediente
    total_atendidos = len(chegadas)

    # Utilização = quanto do tempo disponível cada caixa ficou ocupado
    # Limitado a 1.0 (100%) para evitar distorções numéricas
    utilizacao_media = float(np.mean(
        [min(t / HORAS_DIA, 1.0) for t in tempo_ocupado]
    ))

    # Tamanho médio da fila via Lei de Little: Lq = λ_efetivo × Wq
    # λ_efetivo = clientes realmente atendidos por hora naquele dia
    lambda_ef  = total_atendidos / HORAS_DIA
    fila_media = lambda_ef * espera_media

    return {
        "espera":     espera_media,
        "atendidos":  total_atendidos,
        "utilizacao": utilizacao_media,
        "fila":       fila_media,
    }


# ══════════════════════════════════════════════════════════
# FUNÇÃO: simular
# Chama simular_dia() repetidamente por 30 dias e agrega
# os resultados, calculando médias e totais gerais.
# O sistema é reiniciado a cada dia (sem memória entre dias).
# ══════════════════════════════════════════════════════════
def simular(num_caixas: int) -> dict:

    # Executa a simulação para cada um dos 30 dias úteis
    dias = [simular_dia(num_caixas) for _ in range(DIAS)]

    return {
        # Converte espera de horas → minutos para facilitar leitura
        "espera_min": np.mean([d["espera"]     for d in dias]) * 60,
        # Soma total de clientes nos 30 dias
        "atendidos":  int(np.sum([d["atendidos"]  for d in dias])),
        # Média da utilização diária, em percentual
        "utilizacao": np.mean([d["utilizacao"] for d in dias]) * 100,
        # Média do tamanho médio diário da fila
        "fila":       np.mean([d["fila"]        for d in dias]),
    }


# ══════════════════════════════════════════════════════════
# EXPERIMENTOS
# Rodamos a simulação completa (30 dias) para diferentes
# quantidades de caixas, permitindo comparar como cada
# configuração afeta o desempenho do sistema.
# ══════════════════════════════════════════════════════════
SEP  = "─" * 52
SEP2 = "═" * 52

print(f"\n{SEP2}")
print(f"{'SIMULAÇÃO — SISTEMA DE FILAS  (M/M/c)':^52}")
print(f"  λ = {LAMBDA} cli/h  |  μ = {MU} cli/h/caixa  "
      f"|  {DIAS} dias × {HORAS_DIA}h")
print(SEP2)

for n in [1, 2, 3, 4]:
    r = simular(n)
    label = "caixa" if n == 1 else "caixas"
    print(f"\n  {n} {label}")
    print(f"  {SEP}")
    print(f"  Tempo médio de espera   : {r['espera_min']:>7.2f} min")
    print(f"  Total de clientes       : {r['atendidos']:>7d}")
    print(f"  Utilização média        : {r['utilizacao']:>7.1f} %")
    print(f"  Tamanho médio da fila   : {r['fila']:>7.2f} clientes")

print(f"\n{SEP2}\n")