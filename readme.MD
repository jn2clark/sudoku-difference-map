# Solving Sudoku using iterated projections

<img src="images/sudoku_dm.gif" width="400">


## Alternating Projections
<img src="images/cvx-er.gif" width="520">

Projections onto convex sets (POCS) is a useful method for finding the intersection between convex sets.  A simple example is shown below, where we have two convex constraint sets <img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/d81a84099e7856ffa4484e1572ceadff.svg?invert_in_darkmode" align=middle width=18.23283pt height=22.38159pt/> (red) and <img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/85f3e1190907b9a8e94ce25bec4ec435.svg?invert_in_darkmode" align=middle width=18.23283pt height=22.38159pt/> (blue).  The intersection is found by simply projecting onto each set consecutively via the iterated map:

<p align="center"><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/33ec2cdef775eac131faff5e95e6b440.svg?invert_in_darkmode" align=middle width=136.38207pt height=16.376943pt/></p>

where <img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/df5a289587a2f0247a5b97c1e8ac58ca.svg?invert_in_darkmode" align=middle width=12.788655pt height=22.38159pt/> are the projections onto their respective sets.  Projections are idempotent <img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/1a9e1531579f1943d917c873e3d17644.svg?invert_in_darkmode" align=middle width=60.253215pt height=22.38159pt/> and are distance minimizing;

<img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/1fd7aeccc5b9f88aef6ca6ed7352321b.svg?invert_in_darkmode" align=middle width=98.3103pt height=24.5652pt/> such that <img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/7c698d856efd5c430bba63efab6bb795.svg?invert_in_darkmode" align=middle width=63.531435pt height=24.5652pt/>  is minimized.

The solution is found when
<p align="center"><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/679186591f8b862e4f94d40ee0bff78b.svg?invert_in_darkmode" align=middle width=84.94629pt height=13.656621pt/></p>.
The convexivity of the constraints sets allows lots of general conclusions to be drawn, even if there is little knowledge about the individual sets.  Unfortunately, when the constraint sets are non-convex, very few general results hold. Consequently, rather than finding the global solution, using simple alternating projections can result in stagnation at local minima.  An example of this is shown below, where the sets from the previous example are made to be non-convex, and the ability to find the intersection (global minima) is now highly dependent on the initial guess.

<img src="images/ncvx-er.gif" width="520">

Despite the loss of guarantees when the sets are no longer convex, projection methods can prove very useful and powerful for finding solutions to non-convex optimization problems.  Examples include Sudoku, the n-queens problem , graph-coloring and phase retrieval.

As I will discuss in the next section, to get projection methods to work with non-convex problems, the simple alternating projection algorithm needs to be modified.


##  The Difference Map
One of the most successful non-convex projection algorithms is the difference map (DM).  It is written as
<p align="center"><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/7247ecfb09bc63fdd66fe7d001f25ba8.svg?invert_in_darkmode" align=middle width=150.3645pt height=16.376943pt/></p>
where
<p align="center"><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/a1028b793282df0353b6462bef1574dd.svg?invert_in_darkmode" align=middle width=153.02892pt height=16.376943pt/></p>,
<p align="center"><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/2949f2e3e9a034deb7c0e74c435d49db.svg?invert_in_darkmode" align=middle width=464.13675pt height=16.376943pt/></p>
where <img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/f7019b486d7fc8f840b0ce0bb0d41714.svg?invert_in_darkmode" align=middle width=14.55729pt height=14.10222pt/> and <img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/4c512beeb3e83909b7e19f3cabcfa395.svg?invert_in_darkmode" align=middle width=14.55729pt height=14.10222pt/> are called estimates.
Once a fixed point has been reached,

<p align="center"><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/4ba1122a50f9b529e295422f4eb5c7f3.svg?invert_in_darkmode" align=middle width=129.5481pt height=16.376943pt/></p>

and this implies the two estimates are equal with a solution;
<p align="center"><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/6c81217d679260d5525471e061e85c6c.svg?invert_in_darkmode" align=middle width=100.79949pt height=10.2355935pt/></p>


Rather than the form outlined above, a simpler version of the difference map is often used and is given by;
<p align="center"><img src="https://rawgit.com/jn2clark/sudoku-difference-map/master/svgs/08d602eabb2c0d3cc88ccbd108a45dd1.svg?invert_in_darkmode" align=middle width=271.10655pt height=16.376943pt/></p>.
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

