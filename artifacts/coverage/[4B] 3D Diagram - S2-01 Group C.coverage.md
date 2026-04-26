## Coverage summary: [4B] 3D Diagram - S2-01 Group C.json

- total expressions: **21**
- meshed expression ids: **9**
- supported expressions: **11**
- supported but not meshed: **2**
- recognized unsupported (reserved; classifier uses **0** — see geometry ineligible): **0**
- geometry ineligible (non-mesh: params, inequalities we skip, etc.): **8**
- unrecognized: **2**

## Top missing groups

| missing | total | family | status | fingerprint | example ids | example normalized |
|---:|---:|---|---|---|---|---|
| 1 | 1 | CONSTANT_PLANE | SUPPORTED | `d328cd868bc44953` | `50` | `z=1.2{abs(abs(x)-2.5)<0.3}{y<-3}{y>-3.5}` |
| 1 | 1 | UNKNOWN | UNRECOGNIZED | `2cb67c2f5b247c4e` | `49` | `(abs(x)-2.5)^2+4(z-2.5)^2=0.3^2{z>2.5}{y<-3}{y>-3.5}` |
| 1 | 1 | UNKNOWN | UNRECOGNIZED | `804e6af81665a5a2` | `47` | `abs(abs(x)-2.5)=0.3{y<-3}{y>-3.5}{1.2<z<2.5}` |
| 1 | 1 | Z_SLAB_REGION | SUPPORTED | `1c4639a1e5de379e` | `28` | `-(1.5{abs(x)>1.4)<x<1.5{abs(x)>1.4}{y>-4}{y<-3}{0.5<z<2.5}` |
