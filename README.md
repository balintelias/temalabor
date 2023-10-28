# BME - VIK Villamosmérnök BSc

## Infokomm specializáció, nagyfrekvenciás rendszerek ágazat

### Témalabor a DOCS laborban

Először a [DSPIllustrations](https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html) oldalán található OFDM Python szimulációt néztem át, és értelmeztem.

Kivettem belőle a cyclic prefix-et használó kódrészeket, és módosítottam 128 alvivős QPSK modulációra.
Az example.py tartalmazza az átvett kódot, a simulate.py pedig az SNR függvényében ábrázolja a bithibaarányt (elég göröngyösen még, de az egy további probléma).