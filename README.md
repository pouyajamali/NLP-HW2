# NLP-HW2

 In this homework, we are working on the task of finding a suitable substitution for a target word in a sentence.There is a default solution that produces 10 candidates for each lexical substitution and if any of them match the substitute words preferred by a group of human annotators then that substitution is marked as correct.The overall score reported is the precision score over the entire data set which is described in detail in the Accuracy section below. The accuracy of the default solution is 0.28. What we are going to do is to create a retrofit function which changes the glove.6B.100d.magnitude file so when our solution produces 10 guesses for each lexical substitution the chance of having at least one correct guess would increase. The objective of retrofitting is to learn a matrix Q=(q1,…,qn) such that the columns of the matrix Q are close to use the ontology graph G in order to the word vectors in Q̂ =(q̂ 1,…,q̂ n).The algorithm to find Q is as follows:   
Initialize Q_hat to be equal to the vectors in Q:

 ####lexicon[word] # change neighbours
            qj_word = set(lexicon[word]).intersection(wvVocab) 
            num_neighbor = len(qj_word)

  #### Update Q
            if num_neighbor == 0:
            qj = []
            for w in qj_word:
            qj.append(Q[w])
            qi_hat = Q_hat[word]
                
                
####For iterations t=1…T take the derivative of L(Q) wrt each qi word vector and assign it to zero to get an update: qi=∑j:(i,j)∈Eβijqj+αiq̂ i∑j:(i,j)∈Eβij+αi       

    
                alpha_i = 1
                beta_i_j = 1

                sigma_qj_beta_i_j = 0
                for j in range(len(qj)):
                  sigma_qj_beta_i_j = np.add(sigma_qj_beta_i_j, np.multiply(qj[j], beta_i_j))

                sigma_beta_i_j = 0
                for j in range(len(qj)):
                  sigma_beta_i_j += beta_i_j
                  Q[word] = np.divide(np.add(sigma_qj_beta_i_j, np.multiply(qi_hat, alpha_i)), sigma_beta_i_j + alpha_i)

    return Q
With using this algorith we create the glove.6B.100d.retrofit.magnitude file and use that to find synonyms.
