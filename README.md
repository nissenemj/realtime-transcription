# Reaaliaikainen Transkriptiosovellus

Tämä sovellus mahdollistaa äänen kaappaamisen eri lähteistä ja sen transkription joko reaaliajassa tai pienellä viiveellä. Sovellus käyttää OpenAI:n Whisper-mallia transkriptioon ja tunnistaa eri puhujat.

## Ominaisuudet

- Äänen kaappaus mikrofonista
- Järjestelmän äänen kaappaus (macOS, vaatii BlackHole-ajurin)
- Reaaliaikainen transkriptio
- Puhujien tunnistus ja erottelu
- Tuki useille kielille (suomi, englanti, ruotsi, venäjä, saksa, ranska, espanja)
- Transkription muokkaus suoraan sovelluksessa
- Transkription tallennus tekstitiedostoon

## Näyttökuva

![Sovelluksen näyttökuva](screenshot.png)

## Vaatimukset

- Python 3.8 tai uudempi
- PyTorch
- Transformers
- SoundDevice
- NumPy
- SciPy
- tkinter (yleensä sisältyy Python-asennukseen)

## Asennus

1. Kloonaa repositorio:

```bash
git clone https://github.com/mettenissen/realtime-transcription.git
cd realtime-transcription
```

2. Asenna tarvittavat kirjastot:

```bash
pip install torch transformers sounddevice numpy scipy
```

3. macOS-järjestelmän äänen kaappausta varten asenna BlackHole-ajuri:
   - Lataa ja asenna [BlackHole](https://existential.audio/blackhole/)
   - Määritä järjestelmän ääni kulkemaan BlackHole-laitteen kautta

## Käyttö

1. Käynnistä sovellus:

```bash
python app.py
```

2. Valitse äänilähde pudotusvalikosta
3. Valitse kieli pudotusvalikosta
4. Valitse haluatko käyttää puhujan tunnistusta
5. Paina "Aloita nauhoitus" -painiketta aloittaaksesi äänen kaappauksen ja transkription
6. Paina "Lopeta nauhoitus" -painiketta lopettaaksesi nauhoituksen
7. Transkriptio näkyy tekstialueella ja sitä voi muokata suoraan
8. Voit tyhjentää transkription "Tyhjennä transkriptio" -painikkeella
9. Voit tallentaa transkription "Tallenna transkriptio" -painikkeella

## Järjestelmän äänen kaappaus

### macOS

1. Asenna BlackHole-ajuri: https://existential.audio/blackhole/
2. Avaa Järjestelmäasetukset > Ääni > Ulostulo
3. Valitse "BlackHole 2ch" ulostulolaitteeksi
4. Avaa sovellus, josta haluat kaapata ääntä
5. Valitse sovelluksessa "BlackHole 2ch" ulostulolaitteeksi
6. Valitse transkriptiosovelluksessa "Järjestelmän ääni (macOS)" äänilähteeksi

### Windows

Windows-järjestelmässä voit käyttää "Stereo Mix" -ominaisuutta järjestelmän äänen kaappaamiseen:

1. Avaa Ääniasetukset napsauttamalla hiiren kakkospainikkeella kaiutinkuvaketta tehtäväpalkissa
2. Valitse "Äänilaitteet"
3. Valitse "Tallennuslaitteet"
4. Napsauta hiiren kakkospainikkeella tyhjää aluetta ja valitse "Näytä piilotetut laitteet"
5. Ota käyttöön "Stereo Mix" -laite
6. Valitse transkriptiosovelluksessa "Stereo Mix" äänilähteeksi

## Tiedostot

- `app.py` - Pääsovellus ja käyttöliittymä
- `audio_recorder.py` - Äänen kaappaus ja käsittely
- `transcriber.py` - Transkriptio Whisper-mallilla
- `speaker_diarization.py` - Puhujien tunnistus ja erottelu

## Huomautuksia

- Transkriptio tapahtuu paikallisesti, joten se vaatii riittävästi laskentatehoa
- GPU nopeuttaa transkriptiota huomattavasti
- Sovellus jakaa äänen 5 sekunnin paloihin ja transkriptoi ne erikseen
- Transkription tarkkuus riippuu äänen laadusta ja taustahäiriöistä
- Puhujien tunnistus perustuu hiljaisuuden tunnistukseen ja voi olla epätarkka monimutkaisissa ääniympäristöissä

## Lisenssi

MIT

## Tekijät

- Mette Nissen (@mettenissen)
