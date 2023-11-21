### Taylor's theorem and big-oh notation

#### Course outcomes

| Course outcome | What it is about |
|---|---|
| *Scientific Literacy* | I can explain and give reference to important results about numerical methods and theory in a precise manner with references.|
| *Core Knowledge* (Facts and Intuition) | I can explain important facts about numerical methods and the intuition behind them.|
|*Analysis*| I can give rigorous analysis of numerical methods.|
|*Computation*| I can write programs that implement numerical methods in an efficient and collaborative way, including analyzing and evaluating working programs.|



#### Take-aways

After studying these notes, we will be able to


- use Taylor's theorem in an effective way, (*Analysis*)
	- choose a 'right' order of Taylor polynomial for use (trial and error is okay)
	- check the smoothnees (a.k.a. 'regularity') condition,
	- draw a meaningful conclusion by using the theorem correctly,
- translate big-oh notations to their precise definition, (*Scientific Literacy*)
  - state the definition with help of reference,
  - check whether specific examples satisfy the definition,
- explain what a big-oh notation tells us, (*Core Knowledge*)
  - describe the main tendency of a quantity in an intuitive language,
  - tell which part is negilible and which part matters,
  - use correct big-oh notation after operations.


### Taylor's theorem

###### Taylor's theorem

> **Theorem** (Taylor's theorem with Lagrange remainder)
>
> Suppose $\delta>0$ and a real valued funtion $f$ defined on $I=(x_0-\delta, x_0 +\delta)$ is $k+1$ times differentiable. Then, for any $x\in I$, we have
> \[\begin{split}
> f(x) &= f(x_0) + f'(x_0)(x-x_0)+ \frac{f''(x_0)}{2!}(x-x_0)^2+\cdots
> \\
> &\quad +\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k +\frac{f^{(k+1)}(\xi)}{(k+1)!}(x-x_0)^{k+1},
> \end{split}
> \]
> 
> where $\xi\in (x, x_0)$ if $x < x_0$ or $\xi\in (x_0, x)$ if $x_0 < x$.

[proof of Taylor's theorem with Lagrange remainder 1](https://jhparkyb.github.io/resources/notes/na/pf_TaylorThmLag1_lp3000.png)
[proof of Taylor's theorem with Lagrange remainder 2](https://jhparkyb.github.iodocs/resources/notes/na/pf_TaylorThmLag2_lp3001.png)


### Big-oh notation


#### Motivating examples

###### Example: Big-oh notation with exponential in easy language

 **Example** (Taylor series of exponential -  easy language)

 Recall from calculus, for any $x\in\mathbb{R}$, we have
 $$ e^x = 1+ x + \frac 1 2 x^2 + \frac 1 6 x^3 + \cdots + \frac 1 {n!} x^n +\cdots. $$
 Note that this "equality" is "more true" near $x_0=0$ since the series is expanded around $x_0=0$. For example, if $x=0.1$, then the sum becomes
 $$
e^{0.1} = 1+ 0.1 + \frac {0.01}{2} + \frac {0.001} 6 + \frac {0.0001} {24} + \cdots .
   $$
 	After the first few terms, the magnitude of the remaining terms are so small that it does not change the whole sum much. 

		partial sum	error
	0	1.000000	1.051709e-01
	1	1.100000	5.170918e-03
	2	1.105000	1.709181e-04
	3	1.105167	4.251409e-06
	4	1.105171	8.474231e-08
	5	1.105171	1.408981e-09
	6	1.105171	2.009237e-11
	7	1.105171	2.511324e-13
	8	1.105171	3.108624e-15
	9	1.105171	4.440892e-16
	10	1.105171	4.440892e-16

<details>
<summary>code</summary>

```
#%% 0. Import packages/libraries
import pandas as pd
import numpy as np

#%% 1-a. User parameter settings
#--- parameter settings
x = 0.1
N = 10

#%% 1-b. Preliminary tasks
#--- create array for partial sums
partial_sum = np.zeros((N+1,))

#--- exact value
exact = np.exp(x)

#%% 2-a. Main computation (initial setting/computation)
#--- compute the first term (n=0)
n = 0
nth_term = 1.
partial_sum[n] = nth_term

#%% 2-b. Main computation (remaining computation)
#--- compute partial sums (n=1,2,...,10)
for n in range(1, N+1):
    #--- side note: The line below is equivalent to:
    # nth_term = nth_term * x / n
    nth_term = (x**n)/np.math.factorial(n)
    partial_sum[n] = partial_sum[n-1] + nth_term

#%% 3-a. Post processing of computational results
#--- compute the error of partial sums
err = np.abs(partial_sum - exact)

#%% 3-b. Report (plot, console report, save data, etc.)
#--- organize the result using pandas.DataFrame
columns = ["partial sum", "error"]
df = pd.DataFrame(columns=columns, index=np.arange(N+1))
df['partial sum'] = partial_sum
df['error'] = err

#--- report the result
df.head(N+1)
```
</details>
	
For example, after eye-ball the table above, one may want to focus on the first four terms (the rows number 0 through 3) and thinks "just remember there are some other terms, but they are minor". We denote this as

$$ \begin{split}
e^x &= 1 + x + \frac 1 2 x^2 + \frac 1 6 x^3 + \frac 1 {24} x^4 +\cdots  
	\\
	&=1 + x + \frac 1 2 x^2 + \frac 1 6 x^3 + {\mathcal{O}\!\left( x^4 \right)}.
	\end{split}
$$

The quotation marked part corresponds to the big-oh part.

#### Definition

There are slightly different variants of the definitions. We follow essentially Strichartz's definitions as defined in his excellent book (p. 147) [^2].

[^2]: Strichartz (2000) *The Way of Analysis*, Jones & Bartlett Learning.
###### Definition: Big-oh for finite quantities

> **Definition** (Big oh for finite quantities)
>
> Let $x_0\in\mathbb{R}$ and $f$ and $g$ are real valued functions defined near $x_0$. If there exist (fixed) constants $C>0$ and  $\delta>0$ such that 
> $$|f(x)|\le C |g(x)| \quad \text{for all } x \in (x_0 - \delta, x_0 + \delta),$$
> or,
> $$ \frac{|f(x)|}{|g(x)|}\le C  \quad \text{for all } x\in (x_0 - \delta, x_0 + \delta)\backslash\{x_0\},$$
>  then we write 
	$$
		f(x)={\mathcal{O}\!\left( g(x) \right)} \quad\text{as}\quad x\to x_0,
	$$
>	
> and say *f(x) is a big oh of $g(x)$ near $x_0$* or  *f(x) is of order of $g(x)$ near $x_0$*.
>

###### Definition: Big-oh for growing quantities

>  **Definition** (Big oh for growing quantities)
>
> Let $f$ and $g$ are real valued functions defined on near $\infty$ (i.e., $[N,\infty)$ for some $N\in\mathbb{R} \cup \{-\infty\}$). If there exist (fixed) constants $C>0$ and  $M>0$ such that 
> $$|f(x)|\le C |g(x)| \quad \text{for all } x \in [M,\infty),$$
> <!-- or,
> $$ \frac{|f(x)|}{|g(x)|}\le C  \quad \text{for all } x \in [M,\infty),$$ -->
> then we write
>	$$ f(x)={\mathcal{O}\!\left( g(x) \right)} \quad\text{as}\quad x\to \infty,
	$$
>	
> and say *f(x) is a big oh of $g(x)$ as $x$ grows* or  *f(x) is of order of $g(x)$ as $x$ grows*.

> **Notation** (error form of big-oh)
>
> Let $f,g,h$ be real valued functions defined near $x_0$. If $f(x) - g(x) = {\mathcal{O}\!\left( h(x) \right)}$ as $x\to x_0$, then we write 
> $$ f(x) = g(x) + {\mathcal{O}\!\left( h(x) \right)} \quad\text{as}\quad x\to x_0.$$

#### Interpretation

> **Interpretation** (Controled/bounded by a simpler function)
>
> Suppose $f(x)={\mathcal{O}\!\left( h(x) \right)}$ as $x\to x_0$. To use big-oh notation in a useful way, we usually set:
| $f$ | $h$ |
|---|---|
|function we want to study| a simpler function than $f$ |
> 
> Then, the big-oh condition says that *the magnitude of $f$ can be controlled or bounded by $g$*. 
> 
><!-- > Example (finite quantity)
> $\sin(x) ={\mathcal{O}\!\left( x \right)}$ as $x\to 0$, which is true, means that we can confidently say $\sin(x)$ is smaller than $x$ in absolute value (up to constant multiple) at least near $x_0=0$. 
>
> Example (growing quantity)
> we have $-x^2+3x+\sqrt{x^3}={\mathcal{O}\!\left( x^2 \right)}$ as $x\to \infty$, which is also true. This means $-x^2+3x+\sqrt{x^3}$ is smaller than $x^2$ in magnitude (up to constant multiple) at least for big $x$ values. -->
> **Interpretation** (error form of big-oh)
> 
> The error form is the most common form of big-oh (e.g., $f(x) = g(x) + {\mathcal{O}\!\left( h(x) \right)}$ as $x\to x_0$). Intuitively, we can interpret it as follows:
> - *the error caused by $g(x)$ in place of $f(x)$ is no worse than $h(x)$ (up to constant multiple)*, or
> - *f(x) is made up of $g(x)$ and something that behaves like $h(x)$ (up to constant multiple), which is assumed to be small*.




###### Example: Big-oh notation with cosine
 
>  **Example** (Taylor series of cosine)
> 
>  Recall from calculus, for any $x\in\mathbb{R}$, we have
>  $$
> \cos x=\sum_{n=0}^{\infty} \frac{(-1)^n}{(2 n) !} x^{2 n} =1-\frac{x^2}{2 !}+\frac{x^4}{4 !}-\frac{x^6}{6 !}+\cdots,
> $$
> where the equality is in the sense of infinite sum: the sum on the right hand side converges to the left hand side.  Note that this "equality" is "more true" near $x_0=0$ since the series is expanded around $x_0=0$. For example, if $x=0.1$, then the sum becomes
>  $$
> \cos (0.1)=1-\frac{0.01}{2}+\frac{0.0001}{24}-\frac{0.000001}{720}+\cdots,   $$
>  	After the first few terms, the magnitude of the remaining terms are so small that it does not change the whole sum much. For example, one may want to focus on the first three terms and write
> 
> $$ \begin{split}
> \cos x &= 1 - \frac{x^2}{2 !} + \frac{x^4}{4 !}-\frac{x^6}{6 !}+\cdots	\\
> 	&=1 - \frac{x^2}{2 !} + \frac{x^4}{4 !} + {\mathcal{O}\!\left( x^6 \right)}.
> 	\end{split}
> $$
> 
> The last equality is not trivial because it involves infinitely many terms, but it is true (See [Justification of lumping infinite sum](#justification-of-lumping-infinite-sum)).
>  
<!-- > In words, *the error caused by $1 - \frac{x^2}{2 !} + \frac{x^4}{4 !}$  in place of $\cos(x)$ near $x_0=0$ is essentially $x^6$ (up to a constant multiple), or *near $x_0=0$, $\cos(x)$ behaves almost like $1 - \frac{x^2}{2 !} + \frac{x^4}{4 !}$ with a small quantity similar to $x^6$ (up to constant multiple).* -->

>  **Question**
>
> Give the interpretation of the big-oh notation in the previous example. (Type in a short answer)
>
> (Reminder) This is **about atmosphere**, not getting it right.
> 
> 1. Think for a short time.
> 2. Share your guess with your pair.
> 3. Type your answer in clicker.
> 4. Feel free to say out loud.



###### Example: Big-oh notation for complexity of matrix multiplication
> 
>  **Example** (complexity of matrix multiplication)
> 
> $$\begin{bmatrix}
> 1 & 2 & 3 \\
> 4 & 5 & 6 \\
> 7 & 8 & 9
> \end{bmatrix}
> \begin{bmatrix}
> 2 \\
> 3 \\
> 4 
> \end{bmatrix} = \begin{bmatrix}
> a \\
> b \\
> c 
> \end{bmatrix}
> $$
> 
> Suppose we want to multiply $3\times 3$ matrix and $3\times 1$ column vector. Let us count how many operations are needed to do this. To get $a$, we need 3 (real number) multiplications and 2 (real number) additions. We need the same amount of computation for $b$ and $c$. If we increase the sizes to $4\times 4$ (matrix) and $4\times 1$ (vector), or more generally, to $n\times n$ (matrix) and $n\times 1$ (vector), we need the following number of operations.
| size | ($\times$) | ($+$) | total |
|---|---|---|---|
| 3 | $3\times3=9$| $2\times3=6$ | 15 |
| 4 | $4\times4=16$| $3\times4=12$ | 28 |
| $n$ | ? | ?| ? |
<!--| n | $n\times n=n^2$| $(n-1)\times n=n^2 -n$ | $2n^2 -n$ |

Using the big-oh notation for growing quantities (since $n\to\infty$ as we increase the size), we have
$$ c(n) =  {\mathcal{O}\!\left( n^2 \right)}, $$

where $c(n)$ denotes the number of operations needed to compute the product of $n\times n$ matrix and $n\times 1$ vector. This way, we can have a quick, but still good picture of how much it costs in terms of computation.  
-->
>  **Question**
>
> Fill the table and find the best big-oh notation.
>
| Answer choice | big-oh |
|---|---|
|(A) | ${\mathcal{O}\!\left( n \right)}$ |
|(B) | ${\mathcal{O}\!\left( n^2 \right)}$ |
|(C) | ${\mathcal{O}\!\left( n^3 \right)}$ |
|(D) | I don't see a good answer. |
> 
> (Reminder) This is **about atmosphere**, not getting it right.
> 
> 1. Think for a short time.
> 2. Share your guess with your pair.
> 3. Type your answer in clicker.
> 4. Feel free to say out loud.
 
### Properties

We state and prove some frequently used properties of big-oh relations for the finite quantities(see the boardwork for the proofs). Similar results hold also for the growing quantities.

> **Proposition** (sum)
>
> Suppose $f_1(x) = {\mathcal{O}\!\left( g_1(x) \right)}$, $f_2(x) = {\mathcal{O}\!\left( g_2(x) \right)}$, and $g_1(x) = {\mathcal{O}\!\left( g_2(x) \right)}$ as $x\to x_0$. Then, 
> $$ f_1(x) + f_2(x) = {\mathcal{O}\!\left( g_2(x) \right)} \quad\text{as}\quad x\to x_0.$$

[Proof of sum of big-oh's](https://jhparkyb.github.io/resources/images/na/na_bigoh_p2000.jpg)

[Example of sum of big-oh's](https://jhparkyb.github.io/resources/images/na/na_bigoh_p2001.jpg)

> **Corollary** (sum of the same big-oh terms)
>
> Suppose $f_1(x) = {\mathcal{O}\!\left( g(x) \right)}$ and $f_2(x) = {\mathcal{O}\!\left( g(x) \right)}$ as $x\to x_0$. Then, 
> $$ f_1(x) + f_2(x) = {\mathcal{O}\!\left( g(x) \right)} \quad\text{as}\quad x\to x_0.$$

[Proof of sum of big-oh's: same order](https://jhparkyb.github.io/resources/images/na/na_bigoh_p2002.jpg)


[Example of sum of big-oh's: same order](https://jhparkyb.github.io/resources/images/na/na_bigoh_p2003.jpg)

> **Proposition** (product)
>
> Suppose $f_1(x) = {\mathcal{O}\!\left( g_1(x) \right)}$ and $f_2(x) = {\mathcal{O}\!\left( g_2(x) \right)}$ as $x\to x_0$. Then, 
> $$ f_1(x) f_2(x) = {\mathcal{O}\!\left( g_1(x) g_2(x) \right)} \quad\text{as}\quad x\to x_0.$$

[Proof of product of big-oh's](https://jhparkyb.github.io/resources/images/na/na_bigoh_p2004.jpg)




> **Corollary** (product directly added to order)
>
> Suppose $f(x) = {\mathcal{O}\!\left( g(x) \right)}$ as $x\to x_0$. Then, 
> $$ f(x) h(x) = {\mathcal{O}\!\left( g(x) h(x) \right)} \quad\text{as}\quad x\to x_0.$$

[Proof of product of big-oh's: directly added to order](https://jhparkyb.github.io/resources/images/na/na_bigoh_p2005.jpg)


[Example of product of big-oh's: directly added to order](https://jhparkyb.github.io/resources/images/na/na_bigoh_p2006.jpg)


### Examples

###### Example: Application of Taylor's theorem

> **Example** (Accuracy of difference quotient)
>
> Suppose that $f:\mathbb{R} \to \mathbb{R}$ is continuously differentiable near $x_0\in R$. Then, we have
> \[ \frac{f(x_0 + h) - f(x_0)}{h} = f^\prime(x_0) + {\mathcal{O}\!\left( h \right)} \]
> 
> **Interpretation**
>
> 

[Proof of accuracy of difference quotient](https://jhparkyb.github.io/resources/notes/na/pf_DiffQuotConvRate_lp2000.png)


> **Question**
> $$ \begin{split}
\cos x &=1 - \frac{x^2}{2 !} + \frac{x^4}{4 !} + {\mathcal{O}\!\left( x^6 \right)}\quad\text{as}\quad x\to0.
	\\
	c(n) &= 2n^2 -n = {\mathcal{O}\!\left( n^2 \right)} \quad\text{as}\quad n\to\infty,
	\end{split} 
$$
> 
> Give a second look at the examples of $\cos(x)$ and the complexity of matrix multiplication. Which part of each of those quantities is negligible and which part matters?
>  
>
> (Reminder) This is **about atmosphere**, not getting it right.
> 
> 1. Think for a short time.
> 2. Share your guess with your pair.
> 3. Type your answer in clicker.
> 4. Feel free to say out loud.



### Remarks

###### Remark: Advantage of big-oh

> **Remark** (Advantage of big-oh)
>
> Big-oh notation allows us to forget about unimportant details (i.e., exact behaviors of a function) and to focus on what really matters (i.e., what is the magnitude of the function like) when it is helpful.

###### Remark: Big-oh term may or may not matter

> **Remark** (Big-oh term may or may not matter)
>
> Sometimes big-oh term is the one that matters. This is typically true for growing quanitities. Sometimes, big-oh term is the one that is negligible. This is usually the case for finite quantities. 

###### Remark: Big-oh is about inequality, not equality

> **Warning** (Big-oh is about inequality, not equality)
>
> Consider $f(x)=x^3$ and $g(x)=x^2$ around $x_0=0$. If $x$ is close enough to $x_0=0$, say $\delta = 0.1$ (hence $x\in(-0.1, 0.1)$), we have
> $$ |f(x)|=|x|^3 \le 0.1|x|^2.$$ 
>
> Using big-oh notation, it follows
>
> $$f(x) = {\mathcal{O}\!\left( g(x) \right)},$$
> 
> with the constant $C=0.1>0$ appearing in the definition.
>
> However, we do not have "$g(x) = {\mathcal{O}\!\left( f(x) \right)}$ as $x\to 0$."
>
>
> To see this, observe that it is impossible to choose a fixed constant $C>0$ that satisfies "for all $x\in(-0.1, 0.1)$, $|x|^2 \le C|x|^3$": taking smaller and smaller $x$'s, $C|x|^3$ eventually becomes smaller than $|x|^2$ no matter how big $C$ you choose.
>
> Even if we try different $\delta\neq 0.1$, the situation is not changed. This is because the violation of big-oh condition (i.e., inequality flip) happens due to $x$'s that are close to 0. Thus we conclude
>
> $$g(x) \neq {\mathcal{O}\!\left( f(x) \right)} \quad\text{as}\quad x\to 0.$$
>
> Observe, however, as you can see from the above reasoning, big-oh encapsulates inequalities.


###### Remark: The order of error term depends on the context

> **Remark** (The order of error term depends on the context)
> 
> Consider the following example. If we want to focus on the terms up to 2nd order, we write
> $$ \begin{split}
    e^x & = 1+ x + \frac 1 2 x^2 + \underbrace{\frac 1 6 x^3 + \cdots + \frac 1 {n!} x^n +\cdots}_{=\;{\mathcal{O}\left( |x|^3 \right)}}
    \\
    & = 1+ x + \frac 1 2 x^2 + {\mathcal{O}\left( |x|^3 \right)} \quad\text{as}\quad x\to 0.
    \end{split}
    $$
> However, while continuing working on, we may need more terms. If it turns out we need to include one more term, we write
> $$ 
    e^x  = 1+ x + \frac 1 2 x^2 + \frac 1 6 x^3 +  \mathcal{O}\left( |x|^4 \right) \quad\text{as}\quad x\to 0.
    $$
> Both are true. But one thing may not be useful while the other is. It depends the situation. 



### Appendix

###### Justification of lumping infinite sum

In the [example of Taylor seires of exponential](#example-big-oh-notation-with-exponential), we claimed 
\[
    \frac 1 6 x^3 + \frac 1 {24} x^4 +\cdots = \mathcal{O}\left( |x|^3 \right) \quad\text{as}\quad x\to 0.
\]
Let us justify this. 

(step 1)
We prove:
\[
\frac{g(x)}{x^3} \to C \quad\text{as}\quad x \to 0,
\]
where $C>0$ is some constant and 
\[
    g(x):=\frac 1 6 x^3 + \frac 1 {24} x^4 +\cdots.
\]
for some $C>0$. By the [theorem on power series](#theorem-on-power-seires), 
 \[
    \frac{g(x)}{x^3}=\frac 1 6 + \frac 1 {24} x +\cdots
\]
is continuous on the interval of convergence, which is $(-\infty, \infty)$ in this case. Therefore, 
$$ \lim_{x\to 0} \frac{g(x)}{x^3} = \lim_{x\to 0} \left(\frac 1 6 + \frac 1 {24} x +\cdots \right)= \frac 1 6 + 0 + 0 \cdots = \frac 1 6,
$$

where we use the continuity of functions defined by power series in the second equality (see the [Theorem on power series](#theorem-on-power-seires)). Thus, this proves the (step 1) with $C=1/6$.
(step 2)
Since $g(x)/x^3 \to C$ as $x\to0$, by the definition of the limit, for $\epsilon = 1$, there exists $\delta > 0$ such that, for any $0 < |x|< \delta$, we have
\[
    \left| \frac{g(x)}{x^3} - C \right| < 1.
\]

Multiplying through by $|x|^3$ and using the inverse triangle inequality, i.e. $|a| - |b| \le |a-b|$ for all $a,b\in\mathbb{R}$, we have

\[
   |g(x)| - C|x|^3 \le \left|g(x) - Cx^3 \right| < |x|^3.
\]

Rearranging, we obtain


\[
   g(x) \le |g(x)| \le (C+1)|x|^3 .
\]

Q.E.D.

###### Error term via Taylor theorem

The most common situation where big-oh notation is used is when analyzing errors using Taylor theorem. Suppose that the context of the analysis somehow hints that knowing only the terms up to 2nd order is enough. Suppose you are studying the difference $f(x) - f(a)$ for some $x$ close to $a$. First, take $\delta>0$ small enough so that we can apply [Taylor theorem](#taylors-theorem) on $(a-\delta, a+\delta)$ and $f'''(x)$ is continuous on $[a-\delta, a+\delta]$. (We can always take a smaller $\delta$ if one of these two is not met.) Then, $|f'''(x)|$ attains the maximum $M_0$ on the closed interval $[a-\delta, a+\delta]$ ([Extreme value theorem](#extreme-value-theorem)).  Now, for any $x\in(a-\delta, a+\delta)$, we have, by Taylor's theorem,

\[
	f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!} (x-a)^2  +\underbrace{\frac{f'''(\xi)}{3!}(x-a)^3}
	_{\mathcal{O}\!\left( |x-a|^3 \right)},
\]

for some $\xi\in (x, a)$ or $\xi\in (a, x)$ depending on $x < a$ or $a < x$. Thus, for all $x\in (a-\delta, a+\delta)$, we have
$$\left|\frac{f'''(\xi)}{3!}(x-a)^3 \right|\le C|x-a|^3,$$ where $C=M_0/3!$. Therefore, we conclude
$$
\frac{f'''(\xi)}{3!}(x-a)^3 = 
	{\mathcal{O}\!\left( |x-a|^3 \right)}.
$$

###### Theorem on power seires

[^3]: James Stewart (2012) *Calculus Early Transcendentals*, 7th Edition,  Brooks Cole 

> **Theorem** (Power series)[^3]
> 
> Suppose that $f$ has a Taylor expansion around $x=a$ with a radius of convergence $R>0$ or $R=\infty$, that is,
>	\[
>	f(x)=\sum_{n=0}^\infty c_n (x-a)^n \quad \forall x \in (a-R, a+R).
>	\]
>	Then, it is differentiable, hence continuous. Furthermore, its derivative and integral have the same radius of convergence as $f$ and they are given by
>	\[
>	f'(x)=\sum_{n=0}^\infty (n+1)c_n (x-a)^{n} \quad \forall x \in (a-R, a+R),
>	\]
>
>	\[
>	\int f(x) dx =C+\sum_{n=0}^\infty \frac{c_n}{n+1} (x-a)^{n+1} \quad \forall x \in (a-R, a+R).
>	\]


###### Extreme value theorem

> **Theorem** (Extreme value theorem)
>
> Suppose $f$ is continuous on a closed interval $[a,b]$. Then, it attains the maximum and minimum. That is, there are $x_1,x_2\in[a,b]$ such that $f(x_1)=M$, $f(x_2)=m$, and $m\le f(x)\le M$ for all $x\in[a,b]$. 

> **Question**
>
> What part would you make boldfaced in the Extreme value theorem if you were a professor? Why? If you feel like it, answer the same question to other theorems.
>
> (Reminder) This is **about atmosphere**, not getting it right.
> 
> 1. Think for a short time.
> 2. Share your guess with your pair.
> 3. Type your answer in clicker.
> 4. Feel free to say out loud.

#### Summary

- Big 'oh' notation allows us to focus only on the most important term out of (possibly) infinitely many terms. (See [Advantage of big-oh](#remark-advantage-of-big-oh))
	
	- In some cases, all terms except the big-oh term matter. (See [Example: Big-oh notation with cosine](#example-big-oh-notation-with-cosine)) 
    - In other cases, only the big-oh term matters. (See [Example: Big-oh notation for complexity of matrix multiplication](#example-big-oh-notation-for-complexity-of-matrix-multiplication))
- Big 'oh' is essentially *inequality*. It is not an equality, though its notation uses '$=$' sign. (See [Remark: Big-oh is about inequality, not equality](#remark-big-oh-is-about-inequality-not-equality))

- Eventually, people use big-oh notation at an intuitive level. But it is good to see it works rigorously as well to convince yourself. (See [Appendix](#appendix) for some rigorous approaches)


