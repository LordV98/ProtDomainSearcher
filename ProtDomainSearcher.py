#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Autor: Carlos Vinicius Barros Oliveira
# Nome do Software: ProtDomainSearcher

import os
import subprocess
import urllib.request
import gzip
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO
from Bio.PDB import PDBParser, PPBuilder

def baixar_pfam(pfam_dir):
    os.makedirs(pfam_dir, exist_ok=True)
    pfam_hmm = os.path.join(pfam_dir, "Pfam-A.hmm")
    pfam_dat = os.path.join(pfam_dir, "Pfam-A.hmm.dat")

    if not os.path.exists(pfam_hmm):
        urllib.request.urlretrieve("https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz", pfam_hmm + ".gz")
        with gzip.open(pfam_hmm + ".gz", 'rb') as f_in, open(pfam_hmm, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        subprocess.run(["hmmpress", pfam_hmm], check=True)

    if not os.path.exists(pfam_dat):
        urllib.request.urlretrieve("https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz", pfam_dat + ".gz")
        with gzip.open(pfam_dat + ".gz", 'rb') as f_in, open(pfam_dat, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return pfam_hmm, pfam_dat

def extrair_sequencia(pdb_file):
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    with open(pdb_file, 'r') as handle:
        lines = handle.readlines()
    lines = [line for line in lines if not line.startswith("REMARK 350")]
    estrutura = parser.get_structure("prot", StringIO("".join(lines)))
    ppb = PPBuilder()
    sequencia = ''
    for pp in ppb.build_peptides(estrutura):
        sequencia += str(pp.get_sequence())
    return sequencia

def salvar_fasta(seq, fasta_file="temp_seq.fasta"):
    with open(fasta_file, "w") as f:
        f.write(">sequencia\n" + seq + "\n")

def executar_hmmscan(pfam_db, fasta_file, saida="result.domtbl"):
    subprocess.run(["hmmscan", "--domtblout", saida, pfam_db, fasta_file], check=True)

def carregar_descricoes(path_txt):
    descr = {}
    with open(path_txt) as f:
        atual = ""
        for linha in f:
            if linha.startswith("#=GF AC"):
                atual = linha.strip().split()[-1]
            elif linha.startswith("#=GF DE") and atual:
                nome = linha.strip().replace("#=GF DE ", "")
                descr[atual] = nome
                atual = ""
    return descr

def parsear_domtbl(domtbl_file, descricoes):
    dados = []
    with open(domtbl_file) as f:
        for linha in f:
            if linha.startswith("#"): continue
            partes = linha.strip().split()
            if len(partes) < 19: continue
            dom, ini, fim, evalue = partes[0], int(partes[17]), int(partes[18]), float(partes[6])
            dados.append((dom, ini, fim, evalue, descricoes.get(dom, dom)))
    return pd.DataFrame(dados, columns=["Domínio", "Início", "Fim", "E-value", "Nome Popular"])

def plotar_dominios(seq, df, saida_img):
    fig, ax = plt.subplots(figsize=(18, 0.4 * len(df)), dpi=150)
    ax.set_xlim(0, len(seq))
    ax.set_ylim(0, len(df))
    ax.set_xlabel("Posição")
    ax.set_yticks([])
    cores = plt.cm.get_cmap('tab20', len(df["Domínio"].unique()))
    legenda, mapa = [], {dom: cores(i) for i, dom in enumerate(df["Domínio"].unique())}
    for i, (_, row) in enumerate(df.iterrows()):
        cor = mapa[row["Domínio"]]
        ax.add_patch(plt.Rectangle((row["Início"], i), row["Fim"] - row["Início"], 0.8, color=cor))
        ax.text((row["Início"] + row["Fim"]) / 2, i + 0.4, row["Nome Popular"], ha='center', va='center', fontsize=6)
    for dom, cor in mapa.items():
        legenda.append(mpatches.Patch(color=cor, label=dom))
    ax.legend(handles=legenda, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=6)
    plt.tight_layout()
    plt.savefig(saida_img, dpi=300)

# Execução principal
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python ProtDomainSearcher.py arquivo.pdb")
        sys.exit(1)
    pdb_input = sys.argv[1]
    pfam_db, pfam_desc = baixar_pfam("/tmp/pfam")
    seq = extrair_sequencia(pdb_input)
    salvar_fasta(seq)
    executar_hmmscan(pfam_db, "temp_seq.fasta", "saida.domtbl")
    descricoes = carregar_descricoes(pfam_desc)
    df = parsear_domtbl("saida.domtbl", descricoes)
    df.to_csv("dominios.csv", index=False)
    plotar_dominios(seq, df, "dominios.png")
    print("✔️ Arquivos gerados: dominios.csv, dominios.png")