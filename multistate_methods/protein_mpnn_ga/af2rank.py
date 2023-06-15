# Derived from: https://colab.research.google.com/github/sokrypton/ColabDesign/blob/main/af/examples/AF2Rank.ipynb#scrollTo=UCUZxJdbBjZt
# See also: https://github.com/jproney/AF2Rank

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabdesign import clear_mem, mk_af_model
from colabdesign.shared.utils import copy_dict

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import jax

from datetime import datetime #LH
import random #LH
import tempfile #LH

def tmscore(x,y, tmscore_exec):
  temp_dir= tempfile.TemporaryDirectory()
  now = datetime.now() #LH
  randint= random.randint(0, 100000) #LH
  out_names= [f'{temp_dir}/{now.strftime("%H%M%S")}_{n}_{hash(str(z))}_{randint}.pdb' for n, z in enumerate([x, y])] #LH
  # save to dumpy pdb files
  for n,z in enumerate([x,y]):
    out = open(out_names[n],"w") #LH
    for k,c in enumerate(z):
      out.write("ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n" 
                  % (k+1,"CA","ALA","A",k+1,c[0],c[1],c[2],1,0))
    out.close()
  # pass to TMscore
  output = os.popen(f'{tmscore_exec} {out_names[0]} {out_names[1]}') #LH
  
  # parse outputs
  parse_float = lambda x: float(x.split("=")[1].split()[0])
  o = {}
  for line in output:
    line = line.rstrip()
    if line.startswith("RMSD"): o["rms"] = parse_float(line)
    if line.startswith("TM-score"): o["tms"] = parse_float(line)
    if line.startswith("GDT-TS-score"): o["gdt"] = parse_float(line)
  
  temp_dir.cleanup() #LH
  
  return o

class af2rank:
  def __init__(self, pdb, params_dir, tmscore_exec, chain=None, model_name="model_1_ptm", model_names=None):
    self.args = {"pdb":pdb, "chain":chain,
                 "use_multimer":("multimer" in model_name),
                 "model_name":model_name,
                 "model_names":model_names}
    self.params_dir= params_dir
    self.tmscore_exec= tmscore_exec #LH
    self.reset()

  def reset(self):
    self.model = mk_af_model(protocol="fixbb",
                             use_templates=True,
                             use_multimer=self.args["use_multimer"],
                             use_alphafold=True, use_openfold=False,
                             debug=False,
                             model_names=self.args["model_names"],
                             data_dir=self.params_dir) #LH
    
    self.model.prep_inputs(self.args["pdb"], chain=self.args["chain"])
    self.model.set_seq(mode="wildtype")
    self.wt_batch = copy_dict(self.model._inputs["batch"])
    self.wt = self.model._wt_aatype

  def set_pdb(self, pdb, chain=None):
    if chain is None: chain = self.args["chain"]
    self.model.prep_inputs(pdb, chain=chain)
    self.model.set_seq(mode="wildtype")
    self.wt = self.model._wt_aatype

  def set_seq(self, seq):
    self.model.set_seq(seq=seq)
    self.wt = self.model._params["seq"][0].argmax(-1)

  def _get_score(self):
    score = copy_dict(self.model.aux["log"])

    score["plddt"] = score["plddt"]
    score["pae"] = 31.0 * score["pae"]
    score["rmsd_io"] = score.pop("rmsd",None)

    i_xyz = self.model._inputs["batch"]["all_atom_positions"][:,1]
    o_xyz = np.array(self.model.aux["atom_positions"][:,1])

    # TMscore to input/output
    if hasattr(self,"wt_batch"):
      n_xyz = self.wt_batch["all_atom_positions"][:,1]
      score["tm_i"] = tmscore(n_xyz,i_xyz, self.tmscore_exec)["tms"]
      score["tm_o"] = tmscore(n_xyz,o_xyz, self.tmscore_exec)["tms"]

    # TMscore between input and output
    score["tm_io"] = tmscore(i_xyz,o_xyz, self.tmscore_exec)["tms"]

    # composite score
    score["composite"] = score["ptm"] * score["plddt"] * score["tm_io"]
    return score
  
  def predict(self, pdb=None, seq=None, chain=None, 
              input_template=True, model_name=None,
              rm_seq=True, rm_sc=True, rm_ic=False,
              recycles=1, iterations=1,
              output_pdb=None, extras=None, verbose=True):
    
    if model_name is not None:
      self.args["model_name"] = model_name
      if "multimer" in model_name: 
        if not self.args["use_multimer"]:
          self.args["use_multimer"] = True
          self.reset()
      else:
        if self.args["use_multimer"]:
          self.args["use_multimer"] = False
          self.reset()
  
    if pdb is not None: self.set_pdb(pdb, chain)
    if seq is not None: self.set_seq(seq)

    # set template sequence
    self.model._inputs["batch"]["aatype"] = self.wt

    # set other options
    self.model.set_opt(
        template=dict(dropout=not input_template,rm_ic=rm_ic),
        num_recycles=recycles)
    self.model._inputs["rm_template_sc"][:] = rm_sc
    self.model._inputs["rm_template_seq"][:] = rm_seq
  
    # "manual" recycles using templates
    ini_atoms = self.model._inputs["batch"]["all_atom_positions"].copy()
    for i in range(iterations):
      self.model.predict(models=self.args["model_name"], verbose=False)
      if i < iterations - 1:
        self.model._inputs["batch"]["all_atom_positions"] = self.model.aux["atom_positions"]
      else:
        self.model._inputs["batch"]["all_atom_positions"] = ini_atoms
    
    score = self._get_score()
    if extras is not None:
      score.update(extras)

    if output_pdb is not None:
      self.model.save_pdb(output_pdb)
    
    if verbose:
      print_list = ["tm_i","tm_o","tm_io","composite","ptm","i_ptm","plddt","fitness","id"]
      print_score = lambda k: f"{k} {score[k]:.4f}" if isinstance(score[k],float) else f"{k} {score[k]}"
      print(*[print_score(k) for k in print_list if k in score])
    
    return score
