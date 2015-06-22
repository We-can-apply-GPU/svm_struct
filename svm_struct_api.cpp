/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <armadillo>

#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"

void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}

SAMPLE      read_test_examples(char *file)
{
  std::ifstream fin(file);
  SAMPLE   sample;
  EXAMPLE  *examples;
  std::string line;
  fin >> sample.n;
  std::cout << sample.n << std::endl;
  examples = new EXAMPLE[sample.n];
  for (int z=0; z<sample.n; z++)
  {
    //std::cout << z << std::endl;
    fin >> examples[z].x.id;
    int cnt;
    fin >> cnt;
    examples[z].x.n = examples[z].y.n = cnt;
    std::getline(fin, line);
    for (int i=0; i<cnt; i++)
    {
      std::getline(fin, line);
      line += " ";
      //std::cout << line << std::endl;
      int pre = 0, next;
      for (int j=0; j<69; j++)
      {
        next = line.find(' ', pre);
        //std::cout << line.substr(pre, next-pre) << std::endl;
        examples[z].x.seq[i][j] = std::stof(line.substr(pre, next-pre));
        pre = next+1;
      }
    }
  }
  sample.examples=examples;
  return sample;
}

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  std::ifstream fin(file);
  SAMPLE   sample;
  EXAMPLE  *examples;
  std::string line;
  fin >> sample.n;
  std::cout << sample.n << std::endl;
  examples = new EXAMPLE[sample.n];
  for (int z=0; z<sample.n; z++)
  {
    //std::cout << z << std::endl;
    int cnt;
    fin >> cnt;
    examples[z].x.n = examples[z].y.n = cnt;
    std::getline(fin, line);
    for (int i=0; i<cnt; i++)
    {
      std::getline(fin, line);
      line += " ";
      //std::cout << line << std::endl;
      int pre = 0, next;
      for (int j=0; j<69; j++)
      {
        next = line.find(' ', pre);
        //std::cout << line.substr(pre, next-pre) << std::endl;
        examples[z].x.seq[i][j] = std::stof(line.substr(pre, next-pre));
        pre = next+1;
      }
    }
    std::getline(fin, line);
    line += " ";
    int pre = 0, next;
    for (int i=0; i<cnt; i++)
    {
      next = line.find(' ', pre);
      examples[z].y.seq[i] = std::stoi(line.substr(pre, next-pre));
      pre = next+1;
    }
  }
  sample.examples=examples;
  return sample;
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm)
{
  sm->sizePsi=5616; /* replace by appropriate number of features */
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;
  long     sizePsi=sm->sizePsi;
  long     i;
  WORD     words[2];

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
  else { /* add constraints so that all learned weights are
            positive. WARNING: Currently, they are positive only up to
            precision epsilon set by -e. */
    c.lhs=(DOC**)my_malloc(sizeof(DOC *)*sizePsi);
    c.rhs=(double*)my_malloc(sizeof(double)*sizePsi);
    for(i=0; i<sizePsi; i++) {
      words[0].wnum=i+1;
      words[0].weight=1.0;
      words[1].wnum=0;
      /* the following slackid is a hack. we will run into problems,
         if we have move than 1000000 slack sets (ie examples) */
      c.lhs[i]=create_example(i,0,1000000+i,1,create_svector(words,"",1.0));
      c.rhs[i]=0.0;
    }
  }
  return(c);
}

const int SIZE = 1000000;
struct STATE
{
  int now;
  float score;
  STATE *pre;
}state_memory[SIZE];
int top = 0;

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  LABEL ybar;
  using namespace arma;
  mat obser(x.n, 69), psi_obser(69, 48);
  for (int i=0; i<x.n; i++)
    for (int j=0; j<69; j++)
      obser(i, j) = x.seq[i][j];
  for (int i=0; i<48; i++)
    for (int j=0; j<69; j++)
      psi_obser(j, i) = sm->w[i*69+j+1];
  mat prob = obser * psi_obser;

  for (int i=0; i<x.n; i++)
  {
    int max = 0;
    for (int j=1; j<48; j++)
    {
      if (prob(i, j) > prob(i, max))
        max = j;
    }
    std::cout << max << " ";
  }
  std::cout << std::endl;

  STATE *pre[48], *now[48];
  for (int i=0; i<48; i++)
  {
    pre[i] = state_memory+top;
    top ++;
    if (top == SIZE)
      top = 0;
    pre[i]->now = i;
    pre[i]->score = prob(0, i);
    pre[i]->pre = NULL;
  }
  for (int i=1; i<x.n; i++)
  {
    for (int j=0; j<48; j++)
    {
      now[j] = state_memory+top;
      top ++;
      if (top == SIZE)
        top = 0;
      now[j]->score = -1e50;
      for (int k=0; k<48; k++)
      {
        float new_score = pre[k]->score + prob(i, j) + sm->w[48*69+k*48+j+1];
        if (new_score > now[j]->score)
        {
          now[j]->now = j;
          now[j]->score = new_score;
          now[j]->pre = pre[k];
        }
      }
    }
    for (int j=0; j<48; j++)
      pre[j] = now[j];
  }
  STATE *max = pre[0];
  for (int i=1; i<48; i++)
  {
    if (pre[i]->score > max->score)
      max = pre[i];
  }
  ybar.n = x.n;
  for (int i=x.n-1; i>=0; i--)
  {
    ybar.seq[i] = max->now;
    max = max->pre;
  }
  return ybar;
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar)) */

  LABEL ybar;

  /* insert your code for computing the label ybar here */

  return(ybar);
}


LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{

  /* argmax_{ybar} loss(y,ybar)+psi(x,ybar) */

  LABEL ybar;
  using namespace arma;
  mat obser(x.n, 69), psi_obser(69, 48);
  for (int i=0; i<x.n; i++)
    for (int j=0; j<69; j++)
      obser(i, j) = x.seq[i][j];
  for (int i=0; i<48; i++)
    for (int j=0; j<69; j++)
      psi_obser(j, i) = sm->w[i*69+j+1];
  mat prob = obser * psi_obser;

  STATE *pre[48], *now[48];
  for (int i=0; i<48; i++)
  {
    pre[i] = state_memory+top;
    top ++;
    if (top == SIZE)
      top = 0;
    pre[i]->now = i;
    pre[i]->score = prob(0, i);
    pre[i]->pre = NULL;
  }
  for (int i=1; i<x.n; i++)
  {
    for (int j=0; j<48; j++)
    {
      now[j] = state_memory+top;
      top ++;
      if (top == SIZE)
        top = 0;
      now[j]->score = -1e50;
      for (int k=0; k<48; k++)
      {
        float new_score = pre[k]->score + prob(i, j) + sm->w[48*69+k*48+j+1];
        if (y.seq[i] != j) new_score += 1;
        if (new_score > now[j]->score)
        {
          now[j]->now = j;
          now[j]->score = new_score;
          now[j]->pre = pre[k];
        }
      }
    }
    for (int j=0; j<48; j++)
      pre[j] = now[j];
  }
  STATE *max = pre[0];
  for (int i=1; i<48; i++)
  {
    if (pre[i]->score > max->score)
      max = pre[i];
  }
  ybar.n = x.n;
  for (int i=x.n-1; i>=0; i--)
  {
    ybar.seq[i] = max->now;
    max = max->pre;
  }
  return ybar;
}

int         empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */
  return(0);
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
  /* Returns a feature vector describing the match between pattern x
     and label y. The feature vector is returned as a list of
     SVECTOR's. Each SVECTOR is in a sparse representation of pairs
     <featurenumber:featurevalue>, where the last pair has
     featurenumber 0 as a terminator. Featurenumbers start with 1 and
     end with sizePsi. Featuresnumbers that are not specified default
     to value 0. As mentioned before, psi() actually returns a list of
     SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
     specifies the next element in the list, terminated by a NULL
     pointer. The list can be though of as a linear combination of
     vectors, where each vector is weighted by its 'factor'. This
     linear combination of feature vectors is multiplied with the
     learned (kernelized) weight vector to score label y for pattern
     x. Without kernels, there will be one weight in sm.w for each
     feature. Note that psi has to match
     find_most_violated_constraint_???(x, y, sm) and vice versa. In
     particular, find_most_violated_constraint_???(x, y, sm) finds
     that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
     inner vector product) and the appropriate function of the
     loss + margin/slack rescaling method. See that paper for details. */
  float obser[48][69], trans[48][48];
  for (int i=0; i<48; i++)
    for (int j=0; j<69; j++)
      obser[i][j] = 0;
  for (int i=0; i<48; i++)
    for (int j=0; j<48; j++)
      trans[i][j] = 0;
  for (int i=0; i<x.n; i++)
  {
    for (int j=0; j<69; j++)
    {
      obser[y.seq[i]][j] += x.seq[i][j];
    }
  }
  for (int i=0; i<x.n-1; i++)
  {
    trans[y.seq[i]][y.seq[i+1]] += 1;
  }
  SVECTOR *vec = new SVECTOR();
  vec->words = new WORD[sm->sizePsi+1];
  int index = 0;
  for (int i=0; i<48; i++)
  {
    for (int j=0; j<69; j++)
    {
      if (obser[i][j] == 0)
        continue;
      vec->words[index] = (WORD){i*69+j+1, obser[i][j]};
      index ++;
    }
  }
  for (int i=0; i<48; i++)
  {
    for (int j=0; j<48; j++)
    {
      if (trans[i][j] == 0)
        continue;
      vec->words[index] = (WORD){48*69+i*48+j+1, trans[i][j]};
      index ++;
    }
  }
  vec->words[index] = (WORD){0, 0};
  vec->twonorm_sq = -1;
  vec->userdefined = NULL;
  vec->kernel_id = 0;
  vec->next = NULL;
  vec->factor = 1;
  return vec;
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. sparm->loss_function is set with the -l option. */
  if(sparm->loss_function == 0) { /* type 0 loss: 0/1 loss */
                                  /* return 0, if y==ybar. return 1 else */
  }
  else {
    int loss = 0;
    for (int i=0; i<y.n; i++)
    {
      if (y.seq[i] != ybar.seq[i])
        loss ++;
    }
    return loss;
  }
  return 0;
}

int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
  return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     predicition matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
  if(exnum == 0) { /* this is the first time the function is
		      called. So initialize the teststats */
  }
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* Writes structural model sm to file file. */
  std::ofstream fout("model.dat");
  for (int i=1; i<=sm->sizePsi; i++)
    fout << sm->w[i] << std::endl;
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
  std::ifstream fin(file);
  STRUCTMODEL sm;
  sm.sizePsi = 5616;
  sm.w = new double[sm.sizePsi];
  for (int i=1; i<=sm.sizePsi; i++)
    fin >> sm.w[i];
  return sm;
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
  for (int i=0; i<y.n; i++)
  {
    fprintf(fp, "%d ", y.seq[i]);
  }
  fprintf(fp, "\n");
} 

void        free_pattern(PATTERN x) {
  /* Frees the memory of x. */
}

void        free_label(LABEL y) {
  /* Frees the memory of y. */
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) { 
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
  printf("         --* string  -> custom parameters that can be adapted for struct\n");
  printf("                        learning. The * can be replaced by any character\n");
  printf("                        and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
      case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
  printf("         --* string -> custom parameters that can be adapted for struct\n");
  printf("                       learning. The * can be replaced by any character\n");
  printf("                       and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

