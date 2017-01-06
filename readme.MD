# Solving Sudoku using iterated projections

<img src="images/sudoku_dm.gif" width="400">

## Running
Simply run
`<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/ed7d05e71816240118c3b65157a543ea.svg?invert_in_darkmode" align=middle width=939.76575pt height=157.80831pt/>C_1<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/a718879c0cbca6de63017f3224708128.svg?invert_in_darkmode" align=middle width=63.742965pt height=24.5652pt/>C_2<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/82b1d8292369f9447ff3b0aa38f47ae9.svg?invert_in_darkmode" align=middle width=692.14728pt height=24.5652pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/d6e5695a8a00f5e45dfe18445bf75e96.svg?invert_in_darkmode" align=middle width=136.38207pt height=24.5652pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/d58337d48610f0bf4202bfdfca7f94fe.svg?invert_in_darkmode" align=middle width=30.06993pt height=39.45216pt/>P<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/a01787263a3b927c173f627cbfa0c6d5.svg?invert_in_darkmode" align=middle width=499.59888pt height=22.74558pt/>PP = P<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/de9a032a9815058c40034d9b259bd1c0.svg?invert_in_darkmode" align=middle width=204.37098pt height=22.74558pt/>P(x) = y \in C<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/2d6b4bef0bc89eb7b236d145f01e15ac.svg?invert_in_darkmode" align=middle width=63.49497pt height=22.74558pt/>\parallel x - y \parallel<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/1d383a99a0ab671f99ce5e769bccc933.svg?invert_in_darkmode" align=middle width=187.23408pt height=39.45216pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/1cebd48971ce75b17523ac575a4df710.svg?invert_in_darkmode" align=middle width=84.946125pt height=22.38159pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/787c3c70fcc2e8fd2371494cb356c587.svg?invert_in_darkmode" align=middle width=1825.1805pt height=521.06472pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/0dd1d5404534011c0c62c873a3e7206b.svg?invert_in_darkmode" align=middle width=150.364335pt height=24.5652pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/dccf1c8e42913120e6da3befd117b8c6.svg?invert_in_darkmode" align=middle width=44.69883pt height=22.74558pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/bb770b76d162bea22c3d6f54e8d93874.svg?invert_in_darkmode" align=middle width=153.028755pt height=24.5652pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/f12919543ba3498cef5089f49fc0a706.svg?invert_in_darkmode" align=middle width=4.54905pt height=14.10222pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/3b5b89bc38c81bb91c030838f53f5495.svg?invert_in_darkmode" align=middle width=265.81995pt height=47.64078pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/ff43df63604b0f704e319703268f961e.svg?invert_in_darkmode" align=middle width=44.69883pt height=22.74558pt/>y_1<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/fd92a53167b3c6ae9574071613d555dc.svg?invert_in_darkmode" align=middle width=27.010665pt height=22.74558pt/>y_2<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/439e45344cd3be83764e853e86799788.svg?invert_in_darkmode" align=middle width=385.97823pt height=22.74558pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/255d2b6058e09c2b8464e7cf145ba6d8.svg?invert_in_darkmode" align=middle width=129.547935pt height=24.5652pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/55668ca293720edbb2d9bed00e667a41.svg?invert_in_darkmode" align=middle width=425.13768pt height=45.82116pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/3c24ebf1cb914b139608bdb901acca32.svg?invert_in_darkmode" align=middle width=100.799325pt height=14.10222pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/3caabee4adc33ff54b21e7c9ca1c89ea.svg?invert_in_darkmode" align=middle width=565.29495pt height=85.27299pt/><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/455223a81603547d201253c76021be2f.svg?invert_in_darkmode" align=middle width=271.10688pt height=24.5652pt/>$.
This simpler version generally performs well and reduces the number of projections required per iteration.

The same non-convex problem is shown below but now using the DM algorithm.  Rather than getting trapped in the local minima, the algorithm is able to escape, search more of solution space, and finally converge onto a solution.

<img src="images/ncvx-dm.gif" width="520">

## References

V. Elser, 'Phase retrieval by iterated projections', J. Opt. Soc. Am. A/Vol. 20, No. 1/January 2003

V. Elser, et al. 'Searching with iterated maps' 104 (2), 418-423 (2007)
http://www.pnas.org/content/104/2/418.full.pdf

S. Gravel, V. Elser, "Divide and concur: A general approach to constraint satisfaction". Physical Review E. (2008). 78:036706. http://link.aps.org/doi/10.1103/PhysRevE.78.036706

Luke Russel D, “Relaxed averaged alternating reflections for diffraction imaging” Inverse problems, (2005) 21, 37-50

Bauschke H H, Combettes P L and Luke D R 2003 Hybrid projection–reflection method for phase retrieval
J. Opt. Soc. Am. A 20 1025–34

H.H. Bauschke, P.L. Combettes, and D.R. Luke, "Phase retrieval, error reduction algorithm, and Fienup variants: a view from convex optimization". Journal of the Optical Society of America A. (2002). 19:1334-1345

S. Marchesini, 'A unified evaluation of iterative projection algorithms for phase retrieval',  Review of Scientific Instruments 78 (2007).

