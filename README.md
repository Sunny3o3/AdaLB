# AdaLB
An Adam-type optimizer that combines belief-based scaling with long-term gradient memory to stabilize adaptive updates.

In chapter 4.1, we evaluate the performance of the proposed optimization algorithm through a series of synthetic experiments involving objective functions with varying landscape characteristics.

Experimental results on the Burgers, Poisson, and high-order differential equation benchmarks show that AdaLB achieves more stable convergence and lower final errors on problems with stiff or relatively smooth gradient dynamics. Furthermore, when combined with L-BFGS, AdaLB provides a more effective first-stage optimization trajectory than Adam, leading to improved performance in hybrid optimization settings.

At the same time, the results indicate that the effectiveness of long-term memory is problem-dependent. While larger memory parameters are beneficial for the Burgers and Poisson problems, they are less suitable for the high-order benchmark. This observation highlights that no single configuration is universally optimal, and that the memory parameter \(\gamma\) should be selected according to the underlying optimization characteristics.

From a practical perspective, AdaLB introduces only marginal computational overhead and maintains a memory cost comparable to standard Adam-type optimizers, making it suitable for large-scale applications.
