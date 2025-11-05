# Gradient-Descent-G6 — Resumo rápido

Este repositório contém implementações simples de gradiente (descida e subida) para três funções: f, g e h. É utilizado para o exercício APS2.

Arquivos principais
- `functions.py` — define as derivadas parciais e gradientes:
  - f: `dfx`, `dfy` — derivadas parciais da função f; `f_grad(x,y)` retorna (dfx, dfy).
  - g: `dgx`, `dgy` — derivadas parciais da função g; `g_grad(x,y)` retorna (dgx, dgy).
  - h: `dhx`, `dhy` — derivadas parciais da função h; `h_grad(x,y)` retorna (dhx, dhy).

- `main.py` — implementa os algoritmos de gradient descent e gradient ascent. É chamado pela linha de comando para minimizar (f ou g) ou maximizar (h).

Como executar
1. Executar minimização de `f` (descida de gradiente):

```bash
python3 main.py f <x0> <y0> <learning_rate> <precision>
```

Exemplo:
```bash
python3 main.py f 0 0 0.1 1e-6
```

2. Executar minimização de `g` (descida de gradiente):

```bash
python3 main.py g 0 0 0.01 1e-6
```

3. Executar maximização de `h` (subida de gradiente):

```bash
python3 main.py h 0 0 0.01 1e-6
```

Opção `debug`: se adicionada como último argumento e definida como `true`, imprime iterações.

Exemplo com debug:
```bash
python3 main.py f 0 0 0.1 1e-6 true
```

Referência do exercício
- Veja o relatório da atividade com mais detalhes: [`APS2.md`](/APS2.md) (ou [`APS2.pdf`](/APS2.pdf)). Ele contém enunciado, requisitos e informações adicionais sobre as tarefas.

Notas
- O módulo `functions.py` usa `math.exp` para exponenciais.
- `main.py` detecta automaticamente se deve usar descendente (f e g) ou ascendente (h).

