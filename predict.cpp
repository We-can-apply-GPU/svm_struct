#include <fstream>
#include <iostream>
/*
#ifdef __cplusplus
extern "C" {
#endif
#include "svm_light/svm_common.h"
#include "svm_light/svm_learn.h"
#ifdef __cplusplus
}
#endif
*/
#include "svm_struct_api.h"
#include "svm_struct_api_types.h"

int main(int argc, char *argv[])
{
  if (argc < 2)
    exit(-1);
  std::ofstream fout("data/ans.dat");
  SAMPLE data = read_test_examples("data/test.dat");
  STRUCTMODEL sm = read_struct_model(argv[1], NULL);
  for (int i=0; i<sm.sizePsi; i++)
    std::cout << sm.w[i+1] << std::endl;
  for (int i=0; i<1; i++)
  {
    LABEL ans = classify_struct_example(data.examples[i].x, &sm, NULL);
    for (int i=0; i<ans.n; i++)
    {
      std::cout << ans.seq[i] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
