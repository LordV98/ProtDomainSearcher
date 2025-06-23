# 🔧 Instalação automática
!apt-get install -y hmmer
!pip install -q biopython

# 📂 Upload do .pdb
from google.colab import files
print("🔼 Envie o arquivo .pdb para análise (uma cadeia por vez)")
uploaded = files.upload()
pdb_file = next(iter(uploaded))

# ================= PROTDOMAINSEARCHER FINAL =================
import os, subprocess, urllib.request, gzip, shutil
import pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as mpatches
import numpy as np
from Bio.PDB import PDBParser, PPBuilder, is_aa

def baixar_pfam(pfam_dir="/content/pfam"):
    os.makedirs(pfam_dir, exist_ok=True)
    pfam_hmm = os.path.join(pfam_dir, "Pfam-A.hmm")
    pfam_dat = os.path.join(pfam_dir, "Pfam-A.hmm.dat")
    if not os.path.exists(pfam_hmm):
        urllib.request.urlretrieve("https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz", pfam_hmm + ".gz")
        with gzip.open(pfam_hmm + ".gz", 'rb') as f_in, open(pfam_hmm, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
    if not all(os.path.exists(pfam_hmm + ext) for ext in [".h3f", ".h3i", ".h3m", ".h3p"]):
        subprocess.run(["hmmpress", pfam_hmm], check=True)
    if not os.path.exists(pfam_dat):
        urllib.request.urlretrieve("https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz", pfam_dat + ".gz")
        with gzip.open(pfam_dat + ".gz", 'rb') as f_in, open(pfam_dat, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
    return pfam_hmm, pfam_dat

def extrair_sequencia(pdb_file):
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    estrutura = parser.get_structure("prot", pdb_file)
    ppb = PPBuilder()
    for modelo in estrutura:
        for cadeia in modelo:
            if len(cadeia) > 30:
                seq = ''.join(str(pp.get_sequence()) for pp in ppb.build_peptides(cadeia))
                return seq, cadeia.get_id()
    return '', None

def salvar_fasta(seq, nome="temp_seq.fasta"):
    with open(nome, "w") as f: f.write(">sequencia\n" + seq + "\n")

def executar_hmmscan(banco, fasta, saida="saida.domtbl"):
    subprocess.run(["hmmscan", "--domtblout", saida, banco, fasta], check=True)

def carregar_descricoes(pfam_dat):
    descr = {}; atual = ''
    with open(pfam_dat) as f:
        for linha in f:
            if linha.startswith("#=GF AC"): atual = linha.strip().split()[-1]
            elif linha.startswith("#=GF DE") and atual:
                descr[atual] = linha.strip().replace("#=GF DE ", ""); atual = ''
    return descr

def parsear_domtbl(domtbl, descricoes):
    dados = []
    with open(domtbl) as f:
        for linha in f:
            if linha.startswith("#"): continue
            partes = linha.strip().split()
            if len(partes) < 19: continue
            dom, ini, fim, evalue = partes[0], int(partes[17]), int(partes[18]), float(partes[6])
            dados.append((dom, ini, fim, evalue, descricoes.get(dom, dom)))
    return pd.DataFrame(dados, columns=["Domínio", "Início", "Fim", "E-value", "Nome Popular"])

def plotar_dominios(seq, df, img="dominios.png"):
    fig, ax = plt.subplots(figsize=(18, 0.4 * len(df)), dpi=150)
    ax.set_xlim(0, len(seq)); ax.set_ylim(0, len(df)); ax.set_xlabel("Posição"); ax.set_yticks([])
    cores = plt.cm.get_cmap('tab20', len(df["Domínio"].unique()))
    mapa = {dom: cores(i) for i, dom in enumerate(df["Domínio"].unique())}
    legenda = []
    for i, (_, row) in enumerate(df.iterrows()):
        cor = mapa[row["Domínio"]]
        ax.add_patch(plt.Rectangle((row["Início"], i), row["Fim"]-row["Início"], 0.8, color=cor))
        ax.text((row["Início"] + row["Fim"])/2, i + 0.4, row["Nome Popular"], ha='center', va='center', fontsize=6)
    for dom, cor in mapa.items():
        legenda.append(mpatches.Patch(color=cor, label=dom))
    ax.legend(handles=legenda, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=6)
    plt.tight_layout(); plt.savefig(img, dpi=300); plt.show()

def calcular_distancias(pdb_file, cadeia_id, dominios_df, fasta_seq):
    parser = PDBParser(QUIET=True)
    estrutura = parser.get_structure("prot", pdb_file)
    cadeia = estrutura[0][cadeia_id]
    residuos = [res for res in cadeia if is_aa(res) and "CA" in res]
    posicoes = [res["CA"].coord for res in residuos if "CA" in res]
    resultados = []
    for _, linha in dominios_df.iterrows():
        ini, fim = linha["Início"], linha["Fim"]
        nome = linha["Nome Popular"]
        dominio = linha["Domínio"]
        sub_coords = posicoes[ini-1:fim]
        if len(sub_coords) < 2:
            media = None
        else:
            dists = [np.linalg.norm(sub_coords[i] - sub_coords[j])
                     for i in range(len(sub_coords)) for j in range(i+1, len(sub_coords))]
            media = round(np.mean(dists), 2)
        resultados.append((dominio, nome, fim - ini + 1, media))
    df_resultado = pd.DataFrame(resultados, columns=["Domínio", "Nome Popular", "N Resíduos", "Distância Média (Å)"])
    df_resultado.to_csv("distancias_reais.csv", index=False)
    return df_resultado

def limpar_distribuicao_categorica():
    arquivos_para_remover = ["frequencia_dominio_nome.csv", "distribuicao_categorica.png"]
    for arquivo in arquivos_para_remover:
        if os.path.exists(arquivo):
            os.remove(arquivo)

def download_outputs_visiveis():
    for arq in ["dominios.csv", "distancias_reais.csv", "dominios.png"]:
        if os.path.exists(arq): files.download(arq)

# 🚀 Execução principal
pfam_db, pfam_dat = baixar_pfam()
seq, cadeia_id = extrair_sequencia(pdb_file)
salvar_fasta(seq)
executar_hmmscan(pfam_db, "temp_seq.fasta", "saida.domtbl")
descricoes = carregar_descricoes(pfam_dat)
df = parsear_domtbl("saida.domtbl", descricoes)
df.to_csv("dominios.csv", index=False)
plotar_dominios(seq, df, "dominios.png")
df_dist = calcular_distancias(pdb_file, cadeia_id, df, seq)

from IPython.display import display
display(df)
display(df_dist)

limpar_distribuicao_categorica()
download_outputs_visiveis()
