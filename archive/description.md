# Description of archived files

This document describes the information about the wave equations and the boundary/initial conditions corresponding to the code in this folder.

## `gaussian-1d`

- Wave equation $$\frac{\partial^2 u}{\partial t^2}-a^2\frac{\partial^2 u}{\partial x^2}=0 $$
- Initial condition $$\left.u\right|_{t=0}=e^{-x^2}$$ $$\left.\frac{\partial u}{\partial t}\right|_{t=0}=0$$
- Boundary condition $$\left.\frac{\partial u}{\partial x}\right|_{x=0}=0$$

## `gaussian-2d`

- Wave equation $$\frac{\partial^2 u}{\partial t^2}-a^2\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$$
- Initial condition $$\left.u\right|_{t=0}=e^{-(x^2+y^2)}$$ $$\left.\frac{\partial u}{\partial t}\right|_{t=0}=0$$
- Boundary Condition $$\left.\frac{\partial u}{\partial x}\right|_{x=0}=0$$ $$\left.\frac{\partial u}{\partial y}\right|_{y=0}=0$$

## `gaussian-3d`

Similar to the 1d and 2d case, but with extra dimension.

## `plane-3d`

- Wave equation is 3d. The assumed wave function is $$u(t,x,y,z)=A\sin (\omega t+k_1x+k_2y+k_3z)+A$$
- Initial condition $$\left.u\right|_{t=0}=A\sin (k_1x+k_2y+k_3z)+A$$ $$\left.\frac{\partial u}{\partial t}\right|_{t=0}=\omega A\cos (k_1x+k_2y+k_3z)$$
- Boundary condition not set.

## `Sephere-2d`

- Wave equation is 2d. The assumed wave function is $$u(t,x,y)=\frac{1}{r}\sin (t - \frac{r}{a})$$ $$r = \sqrt{x^2 + y^2}$$
- Initial condition $$u|_{t=0}=
 \frac{1}{r}\sin (-\frac{r}{a}),\quad r\ge 1$$ $$u|_{t=0}=
 1,\quad r\lt 1$$

- Boundary condition
    - reflection $$u|_{x=xmin}=u|_{x=xmax}=0$$ $$u|_{y=ymin}=u|_{y=ymax}=0$$
    - reflection2 $$\left.\frac{\partial u}{\partial x}\right|_{x=xmin}=\left.\frac{\partial u}{\partial x}\right|_{x=xmax}=0$$ $$\left.\frac{\partial u}{\partial y}\right|_{y=ymin}=\left.\frac{\partial u}{\partial y}\right|_{y=ymax}=0$$
    - transmission not set.

## `wave-3d`
To be updated...
