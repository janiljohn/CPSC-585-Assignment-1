void vector_copy(std::vector<int>& v, int a){
    // write your code, pseudocode, thoughts
    // do you agree with the return type?

    int size_snapshot = A.size();
    
    for(int x = 0; x < size_snapshot; ++x) {
      A.push_back(A[x]);
    }


}
