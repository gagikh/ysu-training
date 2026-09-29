# Պատկերներ

Պատկերները նախատեսված են Օր 5–9-ի համար։ Գործարկման օրինակ՝ `sbatch submit.sh template data/camera.bmp`։

| Ֆայլ | Չափ | Ձևաչափ | Աղբյուր | Լիցենզիա |
|---|---|---|---|---|
| `camera.bmp` | 512×512 | 8-bit | Lav Varshney | CC0 |
| `brick.bmp` | 512×512 | 8-bit | CC0Textures, Bricks25 | CC0 |
| `astronaut.bmp` | 512×512 | 24-bit | NASA Great Images | public domain |
| `coffee.bmp` | 600×400 | 24-bit | Rachel Michetti | CC0 |
| `chelsea.bmp` | 451×300 | 24-bit | Stéfan van der Walt | CC0 |
| `rocket.bmp` | 640×427 | 24-bit | SpaceX | public domain |
| `retina.bmp` | 1411×1411 | 8-bit | Wikimedia Commons, «Fundus photograph of normal left eye» | CC0 |
| `uniform.bmp` | 512×512 | 8-bit | սինթետիկ, բոլոր պիքսելների արժեքը 128 է | — |

Առաջին յոթ պատկերը վերցված են [scikit-image](https://scikit-image.org)-ի `skimage.data` մոդուլից, որտեղ նշված են նաև դրանց աղբյուրները և լիցենզիաները։ `retina.bmp`-ն պահված է grayscale ձևաչափով, որպեսզի repo-ի չափը փոքր մնա։

`uniform.bmp`-ն նախատեսված է Օր 7-ի համար. histogram կառուցելիս բոլոր պիքսելներն ընկնում են նույն bin-ում, և atomic-ների մրցակցությունն առավելագույնն է։
