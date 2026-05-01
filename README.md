Description of the homogenization framework for origami metamaterials can be found in:

Li, Xuwen, Amin Jamalimehr, Mathias Legrand, and Damiano Pasini. "Homogenization framework for rigid and non-rigid foldable origami metamaterials." Journal of the Mechanics and Physics of Solids (2026): 106519.

asymptoticHomogenization.py is the numerical implementation of asymptotic homogenization method on the Miura tessellation.

Energy-basedHomogenization.py is the numerical implementation of energy-based homogenization method on the Miura tessellation.

detailedMiura.py is the fully detailed Miura tessellation as a reference model.

The folder 'postProcessing' contains sample results of the above codes, which are effective elastic constants of Miura origami from asymptotic homogenization, energy-based homogenization, and the fully detailed model:

meshConvergence.py plots the mesh convergence results of the two homogenization methods.

initFoldAngle.py plots the variation of effective elastic constants as functions of the initial fold angle. It computes and plots the errors compared to the fully detailed model. Results and errors are compared with the literature.

creaseStiffness.py plots the variation of effective elastic constants as functions of the crease stiffness. Results are compared with the literature.

The folder 'curveCreaseOrigami' contains a demonstration of the homogenization framework applied to a curve-crease origami pattern. Please refer to Appendix G in our paper above for the crease pattern and results.

curvedMiuraAH.py is the numerical implementation of asymptotic homogenization method on the curve-crease origami tessellation.

curvedMiuraEH.py is the numerical implementation of energy-based homogenization method on the curve-crease origami tessellation.
