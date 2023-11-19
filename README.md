# Témalabor a DOCS laborban - OFDM, Python

 BME - VIK Villamosmérnök BSc, Infokommunikciós rendszerek specializáció, nagyfrekvenciás rendszerek ágazat (HVT)

 Témalabor tárgy munkájának félév alatti dokumentálása. A félév során az OFDM (Orthogonal Frequency Division Multiplexing) tübbszörös hozzáférési rendszerrel ismerkedek, illetve Pythonban végzek vele kapcsolatos szimulációkat.


## Első szimulációk

Először a [DSPIllustrations](https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html) oldalán található OFDM Python szimulációt néztem át, és értelmeztem.

Kivettem belőle a cyclic prefix-et használó kódrészeket, és módosítottam 128 alvivős QPSK modulációra.
Az example.py tartalmazza az átvett kódot, a simulate.py pedig az SNR függvényében ábrázolja a bithibaarányt.
A kódból még a csatornaközelítős részeket is kivettem, mert először csak frekvenciafüggetlen csatornán átvitt Gaussi fehérzajjal terhelt OFDM szimuláció volt a feladat.

## Megfelelő bitszám

Amikor ez kész volt, az eddigi kódot módosítottam, hogy az elméleti bithibaarányhoz hasonló eredményeket kapjak (ehhez magas SNR esetén több bitet kellett szimulálni). Ezt egymásba ágyazott for ciklusokkal oldottam meg, ami a későbbiekben problémát fog jelenteni.

Miután az alapokat már értettem, újraírtam az egész szimulációt gyakorlatilag a 0-ról egy megadott blokkvázlatnak megfelelően. Minden megfelelően működött, a frekvenciafüggetlen csatornán az átvitel a vártnak megfelelően függött össze a jel-zaj viszonnyal.

## Többutas terjedés

A következő feladat a többutas terjedés szimulációja volt egy FIR (véges impulzusválaszú) szűrővel, ami a bithibaarányt nagyon elrontotta. A eddigi implementációm itt kezdett hibákat mutatni. A szűrő a sziombólumváltáskor tranziens folyamatokat vezet be az időfüggvénybe, ezért nem lehet "csak úgy" egymás után számolni a gerjesztések időtartománybeli reprezentációjával. Az egész időfüggvényt előre le kell generálni, és utána kezdhetjük az optimalizációt.

A tranziens folyamatokat úgy lehet kiküszöbölni, hogy a szimbólumok között hagyunk egy védőidőt, amikor már a következő szimbólum időfüggvényével gerjesztjük a FIR szűrőt, így mire a szimbólum "kezdődik", vagyis lejár a védőidő, már lezajlott a tranziens. Ezt nevezzük CP-nek (Cyclic Prefix), amely hosszabb kell legyen, mint a FIR szűrő hossza (Nyilván a FIR szűrő hossza az adott terjedési útvonalakkal függ össze, de valamilyen valós esetben erre lehet maximumot adni).

Szóval:
- Ki kell számolni, hogy adott SNR-hez hány bitet kell szimulálni
- Ki kell számolni, hogy ez hány OFDM szimbólumot jelent
- Ennyi OFDM szimbólumot kell generálni
- Az időtartománybeli reprezentációba be kell szúrni a CP-t

