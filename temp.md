# Hierarchical shrinkage with one unbiased and one affine-biased signal

## 1. Model

Two measurements of the same latent quantity $\theta_i$: one on its own scale, one on an unknown affine scale.

$$y_i = \theta_i + \varepsilon_i, \qquad x_i = a + b\,\theta_i + u_i, \qquad \theta_i = \mu + \eta_i$$

$$\varepsilon_i \sim N(0,\sigma_i^2), \qquad u_i \sim N(0,\omega^2), \qquad \eta_i \sim N(0,\tau^2), \qquad i = 1,\dots,N$$

all mutually independent.

| | |
|---|---|
| **Known** | $\sigma_i^2$ (heteroskedastic across $i$), $\tau^2$ |
| **Unknown, unit-level** | $\theta_1,\dots,\theta_N$ |
| **Unknown, global** | $\mu$, $a$, $b$, $\omega^2$ |
| **Priors** | flat on $\mu$ and on $a$ |

### Deviation form

$$y_i - \mu = \eta_i + \varepsilon_i, \qquad x_i - (a + b\mu) = b\,\eta_i + u_i$$

Both signals carry the same $\eta_i$, scaled differently. But $x$'s center $a + b\mu$ contains the unknown $a$, so it is not recoverable — **$x$ can inform the cross-sectional spread of $\theta$, never its level.** Everything in the result follows from this.

Implied marginal moments, which are all the data can see:

$$\text{Var}(y_i) = \tau^2 + \sigma_i^2, \qquad \text{Cov}(x,y) = b\tau^2, \qquad \text{Var}(x) = b^2\tau^2 + \omega^2$$

---

## 2. Conditions, and why each is needed

**(C1) $\varepsilon_i \perp \eta_i$ and $u_i \perp \eta_i$ — noise unrelated to signal.**
This is what makes the bias genuinely *affine*. If $\text{Cov}(u,\theta) \neq 0$, the effective slope is $b + \text{Cov}(u,\theta)/\tau^2$ and the two are inseparable: "bias" becomes a $\theta$-dependent distortion, not a fixed $(a,b)$ to invert, and the debiasing program is ill-posed. It also delivers $\text{Cov}(x,y) = \text{Cov}(x,\theta)$, the identity calibration rests on.

**(C2) $u_i \perp \varepsilon_i$ — the two channels share no error.**
Carries the identification. With $\kappa = \text{Cov}(u_i,\varepsilon_i) \neq 0$,

$$\text{Cov}(x,y) = b\tau^2 + \kappa$$

and the moment system no longer closes; $\hat b$ is off by $\kappa/\tau^2$. Fails in practice when $x$ and $y$ are built from shared raw inputs, overlapping windows, or common preprocessing. If $\kappa$ were known the *posterior* step survives — the added precision becomes $(b^2\sigma_i^2 - 2b\kappa + \omega^2)/(\sigma_i^2\omega^2 - \kappa^2)$ — but calibration does not.

**(C3) $\tau^2$ known.**
The data reveal $b\tau^2$ and $b^2\tau^2 + \omega^2$: two equations, three unknowns $(b, \tau^2, \omega^2)$. Fixing $\tau^2$ closes the system. Without it, $\boldsymbol\theta$ is identified only up to scale — "$x$ has a large scale factor" and "$\theta$ is highly dispersed" are observationally identical. This assumption is load-bearing.

**(C4) Flat priors on $\mu$ and $a$.**
The posterior is proper for any $N \ge 1$: $Q = 0$ forces $\theta_i = y_i$, then $a = \bar x - b\bar\theta$, then $\mu = \bar\theta$ — a unique point. Two flat priors don't break it because $y$ pins the level and $x - b\theta$ pins $a$.

---

## 3. Definitions

$$c = \frac{b^2}{\omega^2} + \frac{1}{\tau^2}, \qquad d_i = \left(\frac{1}{\sigma_i^2} + c\right)^{-1}, \qquad p_i = \frac{d_i}{\sigma_i^2} \in (0,1)$$

- $c$ — precision on $\theta_i$ **not** coming from $y_i$ (the $x$-channel plus the prior)
- $d_i$ — posterior variance of $\theta_i$ given $\mu$
- $p_i$ — share of unit $i$'s total precision coming from $y_i$

Directly from the definitions, $p_i = d_i(1/d_i - c)$, hence

$$\boxed{\;p_i + c\,d_i = 1\;} \tag{I}$$

Identity (I) generates every simplification below.

---

## 4. Derivation

The joint density in $(\boldsymbol\theta, \mu, a)$ is Gaussian — linear in all three, with flat priors — so the mode equals the mean, and marginalizing $\mu, a$ leaves $E[\boldsymbol\theta \mid y,x]$ unchanged. Minimize

$$Q(\boldsymbol\theta,\mu,a) = \sum_i \frac{(y_i - \theta_i)^2}{\sigma_i^2} + \sum_i \frac{(x_i - a - b\theta_i)^2}{\omega^2} + \sum_i \frac{(\theta_i - \mu)^2}{\tau^2}$$

### 4.1 Level parameters

$$\frac{\partial Q}{\partial \mu} = -\frac{2}{\tau^2}\sum_i (\theta_i - \mu) = 0 \quad\Longrightarrow\quad \mu = \bar\theta$$

$$\frac{\partial Q}{\partial a} = -\frac{2}{\omega^2}\sum_i (x_i - a - b\theta_i) = 0 \quad\Longrightarrow\quad a = \bar x - b\,\bar\theta$$

Both means **unweighted**: $\mu$ because the prior $\theta_i \sim N(\mu,\tau^2)$ is homoskedastic in $i$; $a$ because the $u_i$ share the common variance $\omega^2$. Substituting turns both remaining terms into deviations:

$$x_i - a - b\theta_i = (x_i - \bar x) - b(\theta_i - \bar\theta), \qquad \theta_i - \mu = \theta_i - \bar\theta$$

$a$ is now gone permanently. This is the formal statement that $x$ enters only demeaned.

### 4.2 Unit parameters

$$\tfrac12 \frac{\partial Q}{\partial \theta_i} = \frac{\theta_i - y_i}{\sigma_i^2} - \frac{b}{\omega^2}\Big[(x_i - \bar x) - b(\theta_i - \bar\theta)\Big] + \frac{\theta_i - \bar\theta}{\tau^2} = 0$$

Collecting in $\theta_i$ and $\bar\theta$:

$$\frac{\theta_i}{\sigma_i^2} + c\,(\theta_i - \bar\theta) = \frac{y_i}{\sigma_i^2} + \frac{b(x_i - \bar x)}{\omega^2}$$

$$\Longrightarrow\quad \theta_i = d_i\left[\frac{y_i}{\sigma_i^2} + \frac{b(x_i - \bar x)}{\omega^2} + c\,\bar\theta\right] \tag{II}$$

### 4.3 Level, self-consistently

$\bar\theta$ appears on both sides of (II). Average over $i$:

$$\bar\theta\left(N - c\sum_j d_j\right) = \sum_j d_j\left[\frac{y_j}{\sigma_j^2} + \frac{b(x_j-\bar x)}{\omega^2}\right]$$

By (I), $\;N - c\sum_j d_j = \sum_j (1 - c\,d_j) = \sum_j p_j$, so

$$\bar\theta = \frac{1}{\sum_j p_j}\sum_j\left[p_j\,y_j + \frac{b\,d_j}{\omega^2}(x_j - \bar x)\right]$$

### 4.4 Assemble

Using $d_i/\sigma_i^2 = p_i$ and $c\,d_i = 1 - p_i$, equation (II) becomes

$$\theta_i - \bar\theta = p_i\,(y_i - \bar\theta) + \frac{b\,d_i}{\omega^2}(x_i - \bar x)$$

### 4.5 Re-center $x$

Split $\bar\theta$ into its $y$- and $x$-parts. The $x$-part carries coefficient $1 - p_i = c\,d_i$, so it merges into the $x$-term with a shifted center

$$\tilde x = \bar x - c\,\frac{\sum_j d_j (x_j - \bar x)}{\sum_j p_j}$$

Simplify using $\sum_j p_j + c\sum_j d_j = N$:

$$\tilde x \sum_j p_j = \bar x\Big(\underbrace{\textstyle\sum_j p_j + c\sum_j d_j}_{=\,N}\Big) - c\sum_j d_j x_j = \sum_j x_j (1 - c\,d_j) = \sum_j p_j\,x_j$$

The unweighted $\bar x$ cancels identically, and $\tilde x$ becomes the $p$-weighted mean of $x$ — the same weighting the level uses.

### 4.6 Covariance

The Hessian of $\tfrac12 Q$ has blocks

$$H_{\theta\theta} = \Sigma^{-1} + cI, \qquad H_{(\mu,a)} = \text{diag}\!\left(\tfrac{N}{\tau^2},\,\tfrac{N}{\omega^2}\right), \qquad H_{\theta,(\mu,a)} = \left[-\tfrac{\mathbf 1}{\tau^2},\ \tfrac{b\mathbf 1}{\omega^2}\right]$$

The Schur complement is

$$\Sigma^{-1} + cI - \left(\frac{1}{N\tau^2} + \frac{b^2}{N\omega^2}\right)\mathbf 1\mathbf 1^\top = \Sigma^{-1} + c\left(I - \frac{\mathbf 1\mathbf 1^\top}{N}\right)$$

Sherman–Morrison, plus (I) once more, gives the covariance below.

---

## 5. Result

$$\ell = \frac{\sum_j p_j\,y_j}{\sum_j p_j}, \qquad \tilde x = \frac{\sum_j p_j\,x_j}{\sum_j p_j}$$

$$\boxed{\;E[\theta_i \mid y, x] \;-\; \ell \;=\; p_i\left[(y_i - \ell) \;+\; \frac{b\,\sigma_i^2}{\omega^2}\,(x_i - \tilde x)\right]\;}$$

$$\text{Var}(\theta_i \mid y,x) = d_i + \frac{c\,d_i^2}{\sum_j p_j}, \qquad \text{Cov}(\theta_i,\theta_j \mid y,x) = \frac{c\,d_i\,d_j}{\sum_j p_j}$$

Everything is deviations. The posterior deviation from the level is a fraction $p_i$ of a combined deviation, with each signal centered at its own $p$-weighted mean. The level $\ell$ comes from $y$ alone.

---

## 6. Calibration of $b$ and $\omega^2$

### The estimator

Weights $w_i = 1/(\tau^2 + \sigma_i^2)$ — the marginal precision of $y_i$. With $w$-weighted means and moments,

$$\hat b = \frac{\widehat{\text{Cov}}_w(x,y)}{\tau^2}, \qquad \hat\omega^2 = \widehat{\text{Var}}_w(x) - \hat b^2\tau^2$$

### Why covariance, not a regression slope

$$\text{Cov}(x,y) = \text{Cov}(a + b\theta + u,\ \theta + \varepsilon) = b\,\text{Var}(\theta) = b\tau^2$$

term by term: $\text{Cov}(a,\cdot) = 0$ ($a$ constant); $\text{Cov}(b\theta,\theta) = b\tau^2$; $\text{Cov}(b\theta,\varepsilon) = 0$ by (C1); $\text{Cov}(u,\theta) = 0$ by (C1); $\text{Cov}(u,\varepsilon) = 0$ by (C2).

**Noise inflates variances but not covariances.** A regression slope is $\text{Cov}(x,y)/\text{Var}(y)$, and the contamination sits entirely in the denominator $\tau^2 + \overline{\sigma^2}$ — that is the attenuation. Dividing by the *known* $\tau^2$ avoids it. Correlation is worse still: it entangles $b$ with both $\omega^2$ and the $\sigma_i^2$.

### Why these weights

Unit $i$'s contribution $(x_i - \bar x)(y_i - \bar y)$ has mean $b\tau^2$ for every $i$ and variance

$$\text{Var}(x_i)\text{Var}(y_i) + \text{Cov}(x_i,y_i)^2 = (b^2\tau^2 + \omega^2)(\tau^2 + \sigma_i^2) + b^2\tau^4$$

$\sigma_i^2$ enters only through $(\tau^2 + \sigma_i^2)$; the other factors are common. So $w_i \propto 1/(\tau^2+\sigma_i^2)$ up to the additive floor $b^2\tau^4$. The floor matters: even a unit with $\sigma_i^2 = 0$ has a noisy product, because $\eta_i$ and $u_i$ are still random. Weights are therefore **compressed** — best-to-worst ratio roughly $(\tau^2 + \sigma^2_{\max})/\tau^2$, not $\sigma^2_{\max}/\sigma^2_{\min}$. Weighting by $1/\sigma_i^2$ over-concentrates and *raises* the variance of $\hat b$.

Note $w_i \neq p_i$. The former is a marginal precision used to estimate a population moment; the latter is a posterior precision share used for centering. Do not substitute one for the other.

### Why this is the likelihood solution, not a shortcut

With $\tau^2$ known, $\text{Var}(y_i) = \tau^2 + \sigma_i^2$ is free of $(b,\omega^2)$, and those two parameters enter the marginal likelihood only through $\text{Cov}(x,y) = b\tau^2$ and $\text{Var}(x) = b^2\tau^2 + \omega^2$. The map is a bijection, so the marginal likelihood is saturated in $(b,\omega^2)$ and **matching those two moments is the marginal MLE.**

### Why not stationarity in $b$

Differentiating $Q$ in $b$ gives the natural-looking

$$b = \frac{\sum_i (\theta_i - \bar\theta)(x_i - \bar x)}{\sum_i (\theta_i - \bar\theta)^2}$$

Correct as an equation, unusable at the joint mode. This is a Neyman–Scott incidental-parameters problem: the $\hat\theta_i$ are shrunk, hence under-dispersed by exactly the posterior variance,

$$\text{Var}\big(E[\theta \mid y,x]\big) = \tau^2 - \overline{d}$$

so the denominator is too small and $\hat b$ is inflated by $\tau^2/(\tau^2 - \overline{d})$ — a factor that does **not** vanish as $N \to \infty$.

The EM (θ-integrated) version repairs it by using posterior second moments:

$$b = \frac{\sum_i E[(\theta_i - \bar\theta)(x_i - \bar x) \mid y,x]}{\sum_i E[(\theta_i - \bar\theta)^2 \mid y,x]}$$

and this reduces to the moment estimator by two identities:

- **Numerator.** Conditioning on $(y,x)$ makes $x_i$ constant, so the numerator is exactly $\widehat{\text{Cov}}(m, x)$ with $m_i = E[\theta_i \mid y,x]$. By the tower property $\text{Cov}(m,x) = \text{Cov}(\theta,x) = b\tau^2 = \text{Cov}(x,y)$.
- **Denominator.** By the law of total variance, $\ \text{Var}(m) + E[\text{Var}(\theta \mid y,x)] = \text{Var}(\theta) = \tau^2$.

Hence $b = \text{Cov}(x,y)/\tau^2$.

### Diagnostic

$\hat\omega^2 \le 0$ means $\tau^2$ is misspecified — $x$ has less spread than $b^2\tau^2$ implies, and the model has no slack elsewhere to absorb it.

---

## 7. Graphical representation

<svg viewBox="0 0 720 400" xmlns="http://www.w3.org/2000/svg" width="720" font-family="ui-sans-serif, system-ui, sans-serif">
<defs>
<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="#3d3929"/></marker>
<marker id="arA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="#c2603f"/></marker>
</defs>
<text x="26" y="26" font-size="13" font-weight="600" fill="#3d3929">Generative model</text>
<rect x="26" y="132" width="348" height="218" rx="12" fill="none" stroke="#3d3929" stroke-width="1" stroke-dasharray="5 4" opacity="0.45"/>
<text x="362" y="338" font-size="11" text-anchor="end" fill="#6b6558">i = 1 … N</text>
<circle cx="112" cy="78" r="25" fill="none" stroke="#3d3929" stroke-width="1.6"/>
<text x="112" y="84" font-size="16" text-anchor="middle" fill="#3d3929">μ</text>
<rect x="256" y="58" width="94" height="40" rx="8" fill="none" stroke="#c2603f" stroke-width="1.6"/>
<text x="303" y="83" font-size="15" text-anchor="middle" fill="#c2603f">a , b</text>
<circle cx="112" cy="196" r="28" fill="none" stroke="#3d3929" stroke-width="1.6"/>
<text x="112" y="202" font-size="16" text-anchor="middle" fill="#3d3929">θᵢ</text>
<circle cx="68" cy="300" r="28" fill="#3d3929" opacity="0.09"/>
<circle cx="68" cy="300" r="28" fill="none" stroke="#3d3929" stroke-width="1.6"/>
<text x="68" y="306" font-size="16" text-anchor="middle" fill="#3d3929">yᵢ</text>
<circle cx="238" cy="300" r="28" fill="#3d3929" opacity="0.10"/>
<circle cx="238" cy="300" r="28" fill="none" stroke="#3d3929" stroke-width="1.6"/>
<text x="238" y="306" font-size="16" text-anchor="middle" fill="#3d3929">xᵢ</text>
<line x1="112" y1="104" x2="112" y2="163" stroke="#3d3929" stroke-width="1.5" marker-end="url(#ar)"/>
<text x="126" y="139" font-size="11" fill="#6b6558">τ²</text>
<line x1="101" y1="222" x2="79" y2="269" stroke="#3d3929" stroke-width="1.5" marker-end="url(#ar)"/>
<text x="76" y="250" font-size="11" text-anchor="end" fill="#6b6558">σᵢ²</text>
<line x1="134" y1="214" x2="215" y2="282" stroke="#c2603f" stroke-width="1.5" marker-end="url(#arA)"/>
<text x="192" y="242" font-size="11" fill="#6b6558">ω²</text>
<path d="M303,100 C303,180 285,240 258,278" fill="none" stroke="#c2603f" stroke-width="1.5" marker-end="url(#arA)"/>
<text x="316" y="196" font-size="11" fill="#6b6558">affine bias</text>
<circle cx="34" cy="373" r="7" fill="#3d3929" opacity="0.09"/>
<circle cx="34" cy="373" r="7" fill="none" stroke="#3d3929" stroke-width="1.2"/>
<text x="47" y="377" font-size="10.5" fill="#6b6558">observed</text>
<circle cx="123" cy="373" r="7" fill="none" stroke="#3d3929" stroke-width="1.2"/>
<text x="136" y="377" font-size="10.5" fill="#6b6558">latent</text>
<line x1="188" y1="373" x2="216" y2="373" stroke="#c2603f" stroke-width="1.6"/>
<text x="223" y="377" font-size="10.5" fill="#6b6558">contaminated path</text>
<line x1="398" y1="14" x2="398" y2="386" stroke="#3d3929" stroke-width="1" opacity="0.18"/>
<text x="422" y="26" font-size="13" font-weight="600" fill="#3d3929">Where each signal goes</text>
<rect x="432" y="62" width="66" height="34" rx="8" fill="#3d3929" opacity="0.10"/>
<rect x="432" y="62" width="66" height="34" rx="8" fill="none" stroke="#3d3929" stroke-width="1.6"/>
<text x="465" y="85" font-size="15" text-anchor="middle" fill="#3d3929">yᵢ</text>
<rect x="614" y="62" width="66" height="34" rx="8" fill="#3d3929" opacity="0.10"/>
<rect x="614" y="62" width="66" height="34" rx="8" fill="none" stroke="#3d3929" stroke-width="1.6"/>
<text x="647" y="85" font-size="15" text-anchor="middle" fill="#3d3929">xᵢ</text>
<rect x="422" y="168" width="106" height="42" rx="8" fill="#3d3929" opacity="0.06"/>
<rect x="422" y="168" width="106" height="42" rx="8" fill="none" stroke="#3d3929" stroke-width="1.4"/>
<text x="475" y="188" font-size="13" text-anchor="middle" fill="#3d3929">level  ℓ</text>
<text x="475" y="203" font-size="10" text-anchor="middle" fill="#6b6558">from y alone</text>
<rect x="570" y="168" width="122" height="42" rx="8" fill="#3d3929" opacity="0.06"/>
<rect x="570" y="168" width="122" height="42" rx="8" fill="none" stroke="#3d3929" stroke-width="1.4"/>
<text x="631" y="188" font-size="13" text-anchor="middle" fill="#3d3929">deviations</text>
<text x="631" y="203" font-size="10" text-anchor="middle" fill="#6b6558">both signals</text>
<line x1="465" y1="98" x2="470" y2="163" stroke="#3d3929" stroke-width="1.5" marker-end="url(#ar)"/>
<path d="M496,84 C560,84 585,120 600,163" fill="none" stroke="#3d3929" stroke-width="1.5" marker-end="url(#ar)"/>
<line x1="647" y1="98" x2="647" y2="163" stroke="#c2603f" stroke-width="1.5" marker-end="url(#arA)"/>
<path d="M614,80 C560,80 545,120 528,158" fill="none" stroke="#6b6558" stroke-width="1.3" stroke-dasharray="4 4" opacity="0.55"/>
<g opacity="0.75"><line x1="556" y1="102" x2="570" y2="116" stroke="#6b6558" stroke-width="1.8"/><line x1="570" y1="102" x2="556" y2="116" stroke="#6b6558" stroke-width="1.8"/></g>
<text x="536" y="136" font-size="10" text-anchor="middle" fill="#6b6558">a unknown</text>
<path d="M475,212 C475,250 500,258 540,262" fill="none" stroke="#3d3929" stroke-width="1.5" marker-end="url(#ar)"/>
<path d="M631,212 C631,250 615,258 585,262" fill="none" stroke="#3d3929" stroke-width="1.5" marker-end="url(#ar)"/>
<rect x="428" y="278" width="268" height="46" rx="8" fill="none" stroke="#3d3929" stroke-width="1.6"/>
<text x="562" y="300" font-size="13" text-anchor="middle" fill="#3d3929">E[θᵢ | y, x] − ℓ</text>
<text x="562" y="316" font-size="12" text-anchor="middle" fill="#6b6558">= pᵢ [ (yᵢ − ℓ) + (b σᵢ²/ω²)(xᵢ − x̃) ]</text>
<text x="562" y="348" font-size="10.5" text-anchor="middle" fill="#6b6558">pᵢ = share of unit i's precision coming from yᵢ</text>
</svg>

**Left.** $\mu$ and $(a,b)$ sit outside the plate — drawn once, shared by all units. Inside, $\theta_i$ has exactly two children, and only the arrow into $x_i$ carries $(a,b)$. That single structural fact is the whole problem: $x_i$ cannot be read without first inverting the affine map, and the map must be estimated from the cross-section.

**Right.** The asymmetry between channels. $y$ has a path to both the level and the deviations; $x$ has a path only to the deviations, because the flat prior on $a$ severs the level route. $\ell$ built from $y$ alone, both centers under $p$-weights, $x$ entering as $x_i - \tilde x$ — all consequences of that missing edge, not of any estimator choice.

**$d$-separation.** Conditioning on $(y,x)$ leaves $\mu$ unobserved, so the $\theta_i$ remain connected through their common parent. That open path is exactly the $c\,d_i d_j / \sum_j p_j$ posterior covariance.

---

## 8. Reading the result

**Relative weight on the two signals.**

$$\frac{\text{coefficient on }(x_i - \tilde x)}{\text{coefficient on }(y_i - \ell)} = \frac{b\,\sigma_i^2}{\omega^2}$$

$b$ is pure unit conversion, so the *magnitude* of the bias never enters — only $x$'s noise. Since $\omega^2$ is common across units while $\sigma_i^2$ is not, $x$ contributes most where $y$ is weakest. **A badly biased but precise $x$ beats an unbiased noisy one.**

**Precision partition.** Writing $T_i = 1/\sigma_i^2$, $T_u = 1/\omega^2$, $T_0 = 1/\tau^2$ and $\Lambda_i = T_i + b^2T_u + T_0$:

$$\frac{T_i}{\Lambda_i} + \frac{b^2T_u}{\Lambda_i} + \frac{T_0}{\Lambda_i} = 1$$

and $p_i = T_i/\Lambda_i$. The three sources split unit $i$'s total precision. $y$ contributes both level and deviation, so its full share rides on $y_i$; $x$ contributes deviation only, its level opinion falling back to $\ell$; the prior has no level of its own.

**Centers.** Only $y$-shares appear in $\ell$ and $\tilde x$. Units informed mostly by $x$ or by the prior get little say in either — correct, since $x$ cannot locate a level.

**Cross-unit covariance** is the price of estimating the level. It decays as $\sum_j p_j$, the effective number of units carrying level information.

**Limits.** $\omega^2 \to \infty$ or $b \to 0$ recovers plain James–Stein shrinkage of $y$; $\omega^2 \to 0$ gives $\ell + (x_i - \tilde x)/b$; $\sigma_i^2 \to 0$ gives no shrinkage for unit $i$. At $N = 1$: $\ell = y_1$, $x_1 - \tilde x = 0$, so $E[\theta_1] = y_1$ — no cross-section, no information from $x$. $x$ carries $N-1$ degrees of freedom, having spent one on $a$.

---

## 9. Scale invariance: what $b$ and $\omega^2$ actually mean

### $b$ alone means nothing

$b$ carries units of [$x$-units] / [$\theta$-units]. Measure $x$ in basis points instead of percent and $b$ multiplies by 100. It is not a shrinkage factor, not a correlation, and there is no reason for it to sit near 1 unless $x$ and $\theta$ happen to share units.

The model is **exactly invariant** to that choice. Under $x \to \lambda x$:

$$b \to \lambda b, \qquad \omega^2 \to \lambda^2\omega^2 \qquad\Longrightarrow\qquad \frac{b^2}{\omega^2}\ \text{fixed}\ \Longrightarrow\ c, d_i, p_i\ \text{fixed}$$

and $\frac{b\sigma_i^2}{\omega^2}(x_i-\tilde x) \to \frac{1}{\lambda}\cdot\lambda\cdot(\text{same})$. Combined with shift invariance ($x \to x+k$ absorbed by $a$), the entire affine group acts trivially on the posterior.

What *is* meaningful: the **sign** of $b$ (whether $x$ tracks $\theta$ or inverts it, given by $\text{sign}\,\widehat{\text{Cov}}(x,y)$), and the bound $b^2 < \widehat{\text{Var}}(x)/\tau^2$ implied by $\hat\omega^2 > 0$.

$b = 1$ becomes meaningful only when $x$ is *intended* as an estimate of $\theta$ on the same scale. Then $b < 1$ says $x$ compresses true differences, $b > 1$ that it over-disperses them.

### $\omega^2/b^2$ is the object that matters

It is the variance of the debiased reading $(x_i - a)/b$ — **$x$'s error expressed in $\theta$-units** — and it is scale-invariant. Three equivalent readings:

**Reliability.** $\text{Var}((x-a)/b) = \tau^2 + \omega^2/b^2$, of which the real part is

$$R_x = \frac{\tau^2}{\tau^2 + \omega^2/b^2}$$

the fraction of apparent cross-sectional dispersion in $x$ that is genuine signal.

**Attenuation identity.** With $R_y = \tau^2/(\tau^2 + \overline{\sigma^2})$,

$$\text{corr}(x,y) = \sqrt{R_x R_y}$$

the correlation between two noisy measures is the geometric mean of their reliabilities. So the raw $x$-vs-$y$ correlation **understates $x$'s quality**, because it is degraded by $y$'s noise too. Disattenuate:

$$R_x = \frac{\text{corr}(x,y)^2}{R_y}, \qquad \frac{\omega^2}{b^2} = \tau^2\,\frac{1-R_x}{R_x}$$

**Equivalent sample size.** If $\sigma_i^2 = v/T_i$ for some per-observation variance $v$, then $x$ is worth $T_{\text{equiv}} = v\,b^2/\omega^2$ observations of $y$. This converts "$x$ looks informative" into a unit already understood.

### Relation to the naive regression of $y$ on $x$

Fitting $y_i = \alpha + \beta x_i + \text{error}$:

$$\beta = \frac{\text{Cov}(x,y)}{\text{Var}(x)} = \frac{b\tau^2}{b^2\tau^2+\omega^2} = \frac{R_x}{b}$$

$\beta$ **conflates** the scale inversion $1/b$ with the reliability haircut $R_x$. It is neither $b$, nor $1/b$, nor the reliability. Inverting:

$$b = \beta\,\frac{\text{Var}(x)}{\tau^2}, \qquad R_x = \beta\,b = \frac{\beta^2\,\text{Var}(x)}{\tau^2}, \qquad R^2 = \text{corr}^2 = R_x R_y$$

$\alpha = \mu(1-R_x) - \beta a$ mixes the true mean, the level bias, and the reliability — not separately interpretable, and $a$ was never recoverable anyway.

**The trap.** The fitted value is *exactly* the $x$-only posterior mean:

$$\alpha + \beta x_i = E[\theta_i \mid x_i]$$

(slope $\text{Cov}(\theta,x)/\text{Var}(x) = \beta$, through $(\bar x,\bar y)$). It is **already shrunk**. Feeding it into the precision-weighted combination shrinks twice. If you want both signals, use the raw debiased reading $(x_i-a)/b$, which has slope $1/b$, not $\beta$. The regression under-corrects relative to pure debiasing by exactly $R_x$, because it is silently doing the shrinkage at the same time.

---

## 10. Application: backtest vs live Sharpe

$x_i$ = in-sample (backtest) Sharpe, $y_i$ = out-of-sample Sharpe, $\theta_i$ = true Sharpe, $i$ indexes strategies.

### The inputs become obtainable

**$\sigma_i^2$ — free.** $\text{Var}(\widehat{SR}) \approx (1 + SR^2/2)/T_i$ with $T_i$ in years. Straight from OOS track length; heteroskedasticity is exactly what you have. The dependence on $SR$ itself is negligible below $|SR| \sim 1$.

**$\tau^2$ — estimable, which makes C3 non-binding in practice.** Since $\text{Var}(y) = \tau^2 + \overline{\sigma^2}$ and $\overline{\sigma^2}$ is known,

$$\hat\tau^2 = \widehat{\text{Var}}(y) - \overline{\sigma^2}$$

the cross-sectional spread of *true* Sharpes net of OOS noise. Negative means your OOS cross-section is pure noise — no detectable dispersion in skill.

**$\omega^2$ — has a theoretical floor.** An honest backtest over $T_{\text{IS}}$ years would give $\omega^2/b^2 = (1+SR^2/2)/T_{\text{IS}}$. The ratio of actual to floor is the **overfitting multiplier**: a 20-year backtest worth 1.5 years is a factor of 13.

### The structural payoff

**Backtest level inflation is harmless.** The notorious "every backtest looks good" is $a > 0$, absorbed by the flat prior, never touching the answer. The level comes from OOS; the backtest supplies only cross-sectional tilt. That is the correct use of a backtest, and the model enforces it structurally rather than by discipline.

### Where overfitting actually lands

Trying $K_i$ configurations and keeping the best gives roughly

$$x_i \approx \theta_i + \sigma_{\text{IS}}\sqrt{2\ln K_i}$$

The selection premium depends on how many configurations were tried and how noisy the IS estimate is — **not on $\theta_i$**. So overfitting is primarily $a > 0$ with $b \approx 1$; variation in $K_i$ across strategies feeds $\omega^2$. Overfitting is a level-and-variance phenomenon, not a slope one.

$b > 1$ requires the premium to *scale with true quality* — plausible if you abandon unpromising ideas quickly and grind on the live ones, so $K_i$ correlates with $\theta_i$. Forces the other way: internal regularization (shrunk signals or parameters) compresses differences toward $b<1$, and IS-screen truncation attenuates measured $\text{Cov}(x,y)$, pushing $\hat b$ down — usually the dominant bias in real samples.

Caveat: "the optimizer amplifies true edges" *is* $\text{Cov}(u,\theta)\neq0$, violating C1. The model cannot distinguish it from a genuine $b>1$; it absorbs both into $\hat b$. Since $b$ is a nuisance scale, this costs nothing — but don't then read $b$ as a clean scale factor.

**Either way it barely matters.** The posterior is invariant to $b$. Overfitting hurts through $\omega^2/b^2$, not through the slope.

### Worked magnitudes

$\tau = 0.3$, three years OOS so $\sigma^2 \approx 1/3$, giving $R_y = 0.09/0.423 = 0.21$. Observe $\text{corr}(x,y) = 0.20$. Then

$$R_x = \frac{0.04}{0.21} = 0.19, \qquad \frac{\omega^2}{b^2} = 0.09\times\frac{0.81}{0.19} = 0.39, \qquad T_{\text{equiv}} \approx 2.6\ \text{years}$$

With $b=1$, $\text{Var}(x) = 0.48$ and $\beta = 0.09/0.48 = 0.19$: haircut the backtest's cross-sectional edge by ~80%. Check: $b = \beta\,\text{Var}(x)/\tau^2 = 0.19\times0.48/0.09 = 1.0$. **A regression slope far below 1 is not evidence that $b<1$** — it is almost entirely the reliability term.

Note also $\sigma_i \approx 0.58$ against $\tau \approx 0.3$: shrinkage is severe, posterior Sharpes sit close to $\ell$, and most observed cross-sectional spread in *either* signal is noise. And a backtest correlating only $0.20$ with OOS is worth about as much as the three-year live record — the low correlation is mostly the OOS's fault.

### What breaks it here

**Selection is the dominant threat.** If your $N$ strategies passed an IS screen, you have conditioned on large $x_i$. Then $\theta_i$ is no longer a draw from $N(\mu,\tau^2)$, $u_i$ is truncated, and both $\hat b$ and $\hat\omega^2$ are biased toward the selection boundary. You need the full set tested, discards included. This matters more than everything else here.

**Cross-sectional correlation.** Strategies in a family share factor exposure, so $\varepsilon_i$ and $\eta_i$ are correlated across $i$. Effective $N$ is the number of independent bets, not rows — which inflates the apparent precision of $\ell$, $\hat b$ and $\hat\tau^2$. Cluster by family, or use the matrix form with non-diagonal $\Sigma$.

**Unequal backtest lengths.** If $T_{\text{IS}}$ varies, $\omega_i^2$ is not common; the $\partial Q/\partial a$ condition picks up $\omega$-weights and $\bar x$ becomes $\omega$-weighted. The derivation survives; the clean unweighted cancellation does not.

**OOS must be genuinely untouched.** If you looked at OOS results and revised anything, $\varepsilon_i$ inherits the selection in $u_i$, $\kappa \neq 0$, and C2 fails — the one violation that kills calibration outright.

---

## 11. Practical cautions

- **Shrink once.** Do not feed fitted values from a regression of $y$ on $x$ into this — those are already posterior means, and you would shrink twice.
- **Plug-in uncertainty.** Treating $\hat b, \hat\omega^2$ as known understates posterior variance. Immaterial at large $N$ (they converge at $\sqrt N$ while $\theta_i$ never converges); estimate jointly in a small cross-section.
- **Fat tails.** $\widehat{\text{Cov}}$ is a product of deviations, so a few joint outliers dominate it regardless of weighting. Winsorize $x$ and $y$ before calibration — this buys more than optimal weighting does.
- **Panels.** With drifting $a_t, b_t$, run calibration within each cross-section, never pooled. A single intercept and slope cannot absorb time-varying bias.
- **Absolute scale.** $x$ can never inform the level. If you need one, it must come from $y$ or an external anchor.
