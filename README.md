Download Link: https://assignmentchef.com/product/solved-cs771a-homework2
<br>
(Second-Order Optimization for Logistic Regression) Show that, for the logistic regression model (assuming each label <em>y<sub>n </sub></em>∈ {0<em>,</em>1}, and no regularization) with loss function

exp(<em>w</em><sup>&gt;</sup><em>x</em><em><sub>n</sub></em>))), iteration <em>t </em>of a <em>second-order </em>optimization based update <em>w</em><sup>(<em>t</em>+1) </sup>= <em>w</em><sup>(<em>t</em>) </sup>− <strong>H</strong><sup>(<em>t</em>)−1</sup><em>g</em><sup>(<em>t</em>)</sup>, where <strong>H </strong>denotes the Hessian and <em>g </em>denotes the gradient, reduces to solving an <em>importance-weighted </em>regression problem of the form <em>w</em><sup>(<em>t</em>+1) </sup>= argmin<em><sub>w</sub></em>, where <em>γ<sub>n </sub></em>denotes the importance of the <em>n<sup>th </sup></em>training example and <em>y</em>ˆ<em><sub>n </sub></em>denotes a modified real-valued label. Also, clearly write down the expression for both, and provide a brief justification as to why the expression of <em>γ<sub>n </sub></em>makes intuitive sense here.

<h2>Problem 2</h2>

(Perceptron with Kernels) We have seen that, due to the form of Perceptron updates <em>w </em>= <em>w </em>+ <em>y<sub>n</sub></em><em>x</em><em><sub>n </sub></em>(ignore the bias <em>b</em>), the weight vector learned by Perceptron can be written as <em>w </em>, where <em>α<sub>n </sub></em>is the number of times Perceptron makes a mistake on example <em>n</em>. Suppose our goal is to make Perceptron learn nonlinear boundaries, using a kernel <em>k </em>with feature map <em>φ</em>. Modify the standard Perceptron algorithm to do this. In particular, for this kernelized variant of the Perceptron algorithm (1) Give the initialization, (2) Give the mistake condition, and (3) Give the update equation.

<h2>Problem 3</h2>

(SVM with Unequal Class Importance) Sometimes it costs us a lot more to classify negative points as positive than positive points as negative. (for instance, if we are predicting if someone has cancer then we would rather err on the side of caution (predicting “yes” when the answer is “no”) than vice versa). One way of expressing this in the support vector machine model is to assign different costs to the two kinds of mis-classification. The primal formulation of this is:

min <em>w</em><em>,b,</em><em>ξ</em>

subject to <em>y<sub>n</sub></em>(<em>w</em><em><sup>T </sup>x</em><em><sub>n </sub></em>+ <em>b</em>) ≥ 1 − <em>ξ<sub>n </sub></em>and <em>ξ<sub>n </sub></em>≥ 0, ∀<em>n</em>.

The only difference is that instead of one cost parameter <em>C</em>, there are two, <em>C</em><sub>+1 </sub>and <em>C</em><sub>−1</sub>, representing the costs of misclassifying positive examples and misclassifying negative examples, respectively.

Write down the Lagrangian problem of this modified SVM. Take derivatives w.r.t. the primal variables and construct the dual, namely, the maximization problem that depends only on the dual variables <em>α</em>, rather than the primal variables. In your final PDF write-up, you need not give each of every step in these derivations (e.g., standard steps of substituting and eliminating some variables) but do write down the key steps. Explain (intuitively) how this differs from the standard SVM dual problem; in particular, how the <em>C </em>variables differ between the two duals.

<h2>Problem 4</h2>

(SGD for <em>K</em>-means Objective) Recall the <em>K</em>-means objective function: .

As we have seen, the <em>K</em>– means algorithm minimizes this objective by taking a greedy iterative approach of assigning each point to its closest center (finding the <em>z<sub>nk</sub></em>’s) and updating the cluster means . The standard <em>K</em>-means algorithm is a batch algorithm and uses all the data in every iteration. It however can be made online by taking a random example <em>x<sub>n </sub></em>at a time, and then (1) assigning <em>x<sub>n </sub></em>“greedily” to the “best” cluster, and (2) updating the cluster means using SGD on the objective L. Assuming you have initialized randomly and are reading one data point <em>x<sub>n </sub></em>at a time,

<ul>

 <li>How would you solve step 1?</li>

 <li>What will be the SGD-based cluster mean update equations for step 2? Intuitively, why does the update equation make sense?</li>

 <li>Note that the SGD update requires a step size. For your derived SGD update, suggest a good choice of the step size (and mention why you think it is a good choice).</li>

</ul>

<h2>Problem 5</h2>

(Kernel <em>K</em>-means) Assuming a kernel <em>k </em>with an infinite dimensional feature map <em>φ </em>(e.g., an RBF kernel), we can neither store the kernel-induced feature map representation of the data points nor can store the cluster means in the kernel-induced feature space. How can we still implement the kernel <em>K</em>-means algorithm in practice? Justify your answer by sketching the algorithm, showing all the steps (initialization, cluster assignment, mean computation), clearly giving the mathematical operations in each. In particular, what is the difference between how the clusters means would need to be stored in kernel <em>K</em>-means versus how they are stored in standard <em>K</em>means? Finally, assuming each input to be <em>D</em>-dimensional in the original feature space, and <em>N </em>to be the number of inputs, how does kernel <em>K</em>-means compare with standard <em>K</em>-means in terms of the cost of input to cluster mean distance calculation (please answer this using the big O notation)?

<h2>Problem 6 (Programming Problem,</h2>

Part 1: You are provided a dataset in the file binclass.txt. In this file, the first two numbers on each line denote the two features of the input <em>x</em><em><sub>n</sub></em>, and the third number is the binary label <em>y<sub>n </sub></em>∈ {−1<em>,</em>+1}.

Implement a generative classification model for this data assuming Gaussian class-conditional distributions of the positive and negative class examples to be and N(<em>x</em>|<em>µ</em><sub>−</sub><em>,σ</em><sub>−</sub><sup>2 </sup><strong>I</strong><sub>2</sub>), respectively. Note that here <strong>I</strong><sub>2 </sub>denotes a 2 × 2 identity matrix. Assume the class-marginal to be <em>p</em>(<em>y<sub>n </sub></em>= 1) = 0<em>.</em>5, and use MLE estimates for the unknown parameters. Your implementation need not be specific to two-dimensional inputs and it should be almost equally easy to implement it such that it works for any number of features (but it is okay if your implementation is specific to two-dimensional inputs only).

On a two-dimensional plane, plot the examples from both the classes (use red color for positives and blue color for negatives) and the learned decision boundary for this model. Note that we are not providing any separate test data. Your task is only to learn the decision boundary using the provided training data and visualize it.

Next, repeat the same exercise but assuming the Gaussian class-conditional distributions of the positive and negative class examples to be N(<em>x</em>|<em>µ</em><sub>+</sub><em>,σ</em><sup>2</sup><strong>I</strong><sub>2</sub>) and N(<em>x</em>|<em>µ</em><sub>−</sub><em>,σ</em><sup>2</sup><strong>I</strong><sub>2</sub>), respectively.

Finally, try out an SVM classifier (with linear kernel) on this data (we’ve also provided the data in the format libSVM requires) and show the learn decision boundary. For this part, you do not need to implement SVM. There are many nice implementations of SVM available, such as <a href="http://scikit-learn.org/stable/modules/svm.html">the one in scikit-learn</a> and the very popular <a href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/">libSVM</a> toolkit. Assume the “C” (or <em>λ</em>) hyperparameter of SVM in these implementations to be 1.

Part2: Repeat the same experiments as you did for part 1 but now using a different dataset binclassv2.txt. Looking at the results of both the parts, which of the two models (generative classification with Gaussian classconditional and SVM) do you think seems to work better for each of these datasets, and in general?

Deliverables: Include your plots (use a separate, appropriately labeled plot, for each case) and experimental findings in the main writeup PDF. Submit your codes in a separate zip file on the provided Dropbox link. Please comment the code so that it is easy to read and also provide a README that briefly explains how to run the code. For the SVM part, you do not have to submit any code but do include the plots in the PDF (and mention the software used – scikit-learn or libSVM).