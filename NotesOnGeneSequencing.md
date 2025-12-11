# Notes on Nucleotides, Sequences, and Drug Targets

## DNA vs RNA

DNA: Adenine(A), Cytosine (C), Guanine(G), Thymine(T)
RNA: Same but swap Thymine for Uracil(U)

## IUPAC ambiguity codes
| Code | Meaning     | Possible Bases |
|------|-------------|----------------|
| **R** | Purine      | A or G         |
| **Y** | Pyrimidine  | C or T         |
| **S** | Strong      | G or C         |
| **W** | Weak        | A or T         |
| **K** | Keto        | G or T         |
| **M** | Amino       | A or C         |
| **B** | Not A       | C, G, T        |
| **D** | Not C       | A, G, T        |
| **H** | Not G       | A, C, T        |
| **V** | Not T       | A, C, G        |
| **N** | Any base    | A, C, G, T     |

## NCBI Viral Data set (Example: Influenza A)
In NCBI viral dataset, 
- Metadata about the virus and its gene segment  
- Collection/source info  
- Publication info  
- The actual nucleotide sequence at the bottom, labeled **`ORIGIN`**

Example for Influenza A Segment

- NCBI record
https://www.ncbi.nlm.nih.gov/nuccore/PV659824.1

On the right-hand panel NCBI lets you run **BLAST** to compare sequences.  
The **FASTA** format here comes from the original **FASTA** alignment software (often back-explained as “Fast-All”).

## ORIGIN Block Format

ORIGIN   
| Position | Sequence                                                    |
|----------|--------------------------------------------------------------|
| 1        | agcraaagca ggcaaaccat ttgaatggat gtcaatccga ctttactttt cttgaaagtt |
| 61       | ccagcgcaga atgccataag caccacattc ccatatactg gagatcctcc atacagccat |
| 121      | ggaacaggaa caggatacac catggataca gtcaacagaa cgcatcaata ctcagaaaaa |
| 181      | ggaaaatgga caacaaacac ggaaactgga gcaccacaac ttaatccaat tgatggacca |
...

These are encoded DNA/RNA sequences with fixed width blocks of 10 bases grouped into lines of 60 bases (nucleotides). On the left position is 1 indexed (most programming languages use base 0 so subtract 1 when reading.)

## Codons and Amino Acids

- A **codon** is a sequence of **3 nucleotides**
- This is the basic “word” in the genetic code
- Each codon corresponds to:
  - a specific **amino acid**, or  
  - a **start**/**stop** signal for translation

There are:

- 4 possible bases per position  
- Codon space: \( 4^3 = 64 \) possible codons  
- These encode:
  - 20 common amino acids  
  - Start/stop signals  

Useful refrences: 
- Nucleic Acids -> amino acids overview :  https://www.nature.com/scitable/topicpage/nucleic-acids-to-amino-acids-dna-specifies-935/#:~:text=In%20fact%2C%20even%20two%20nucleotides,would%20be%20a%20triplet%20code.

-Codon Wheel : https://www.sigmaaldrich.com/US/en/technical-documents/technical-article/genomics/sequencing/amino-acid-codon-wheel

So the pipeline is:

> **Nucleotide sequence → codons → amino acids → proteins**

## Protein Categories (Very High-Level)
then your proteins can be 

**Structural** : Actin, Tubulin, Collagen, Keratin,

**Enzymes** : Plolymerases, kinases, ATP syntesis, Restriction enzymes

**Membrane Proteins**: Ion channels (move ions across cell membrane), Transporters (move sugars, amino acids, nutrients, drugs) -- GLUT, ABC Transporters and then Pumps — e.g. **Na⁺/K⁺ ATPase**

Receptors: GPCRs, Tyrosine Kinase receptors, cytokine receptors, toll-like receptors,

## Viral Drug Targets 

For viruses you're looking for the following as drug targets.  

**strucural** : Capsid proteins, nucleoproteins, membrane proteins, spike proteins, 

**polymerases** (turn on protein synthesis cascades): viral RNA dependant polymerase, reverse transcriptase for retroviruses,

**proteases** -- create polyproteins. 

## Example: Influenza A and Oseltamivir
In this case if we stick with the influenza example, you could use like Oseltamivir (Tamiflu)  -- CHEMBL has all the data on this molecule @ 

https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL1229

SMILE representation (structure as text)

    CCOC(=O)C1=C[C@@H](OC(CC)CC)[C@H](NC(C)=O)[C@@H](N)C1

But then you could click on its targets, and chembl gives you a list of viral targets 

https://www.ebi.ac.uk/chembl/explore/targets/STATE_ID:ac7hWY0imjl-h7u9-qtd9A%3D%3D

So in this case, now you could do something like train a model that does 


Predict nucleotides as a function of SMILE : SMILE Representation -> Nucleotide sequence

Predict Chemical structure as a function of nucleotide sequence:  Nucleotide sequence -> SMILE representation 

get a bi-directional model going. 

## Some data structures 
Base structure is an array for the full text sequence

K-mers / n-grams (sub sequence windows): 3-mers, 5-mers, 9-mers but these are sliding windows, can use hash based indexing. Good for sequence alignment and motif discovery. Protein family classification 

Position specific scoring matrices (PSSM)
matrix of shape Lx20 giving prob of each amino acid at each position -- built from multiple sequence alignments (MSA) -- heavy use in evolutionary modeling and structure prediction.

Seq1: M--TNP-KPQRKTK--
Seq2: MSTTNP-KPQR-TK-
Seq3: M-TTNP-KPQRSTK-P

MSAs are the data structure for AlphaFold, ESMFold...

Protein 3D strcuture data structure 

PDB format (Protein Data Bank) -- rows of atoms with atom ID, residue name, xyz coords, chain id.

Atom Graphs : nodes are atoms, edges are bonds or distance threasholds, and then features are charge, atom type, aromaticity -- GNN like graphformer and GVP. 

Residue Graphs: nodes are amin acids, edges are proximity <8 angstromgs so back bone links, side chain orientation -- protein- protein interaction pred, and fold classification 

Distance maps / contact maps -- LxL matrix where each entry is distance beteween residues 

Torsion angle sequences -- represent protein as a sequence of rotations, phi, gamma, and chi. 