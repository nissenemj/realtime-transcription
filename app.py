import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import platform
import threading
import queue
import time
import os
import sounddevice as sd
from audio_recorder import AudioRecorder
from transcriber import Transcriber

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reaaliaikainen Transkriptio")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)

        # Set up the audio recorder
        self.recorder = AudioRecorder(callback=self.on_audio_chunk, chunk_duration=5)

        # UI variables
        self.recording = False
        self.selected_device = tk.StringVar()
        self.selected_language = tk.StringVar(value="fi")
        self.use_diarization = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Valmis aloittamaan")
        self.transcription_text = ""
        self.ui_update_queue = queue.Queue()

        # Set up the transcriber
        self.transcriber = Transcriber(
            callback=self.on_transcription,
            use_diarization=self.use_diarization.get()
        )

        # Start the transcription processing
        self.transcriber.start_processing()

        # Create the UI
        self.create_ui()

        # Testaa tekstialueen päivitystä heti alussa
        self.root.after(2000, self._test_text_update)

        # Start the UI update thread
        self.update_ui()

    def _test_text_update(self):
        """Test text area update."""
        print("Testataan tekstialueen päivitystä...")
        test_text = "Tämä on testi. Jos näet tämän tekstin, tekstialueen päivitys toimii."

        # Päivitä suoraan
        try:
            self.transcription_text += test_text + "\n\n"
            self.transcription_area.delete(1.0, tk.END)
            self.transcription_area.insert(tk.END, self.transcription_text)
            self.transcription_area.see(tk.END)
            print("Tekstialue päivitetty suoraan testiä varten")
        except Exception as e:
            print(f"Virhe tekstialueen päivityksessä: {e}")

    def create_ui(self):
        """Create the user interface."""
        # Configure style for better visibility
        style = ttk.Style()

        # Set a theme that works well on macOS
        if platform.system() == 'Darwin':  # macOS
            style.theme_use('aqua')
        else:
            try:
                style.theme_use('clam')  # A good fallback theme
            except:
                pass  # Use default theme if clam is not available

        # Configure button style with better contrast
        style.configure('TButton', foreground='black', background='#d9d9d9', font=('TkDefaultFont', 10, 'bold'))
        style.map('TButton',
                 foreground=[('active', 'black'), ('pressed', 'black')],
                 background=[('active', '#ececec'), ('pressed', '#c1c1c1')])

        # Configure label style
        style.configure('TLabel', foreground='black', background='#f0f0f0')

        # Configure frame style
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', foreground='black', background='#f0f0f0')

        # Main frame
        main_frame = ttk.Frame(self.root, padding=10, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control frame (top)
        control_frame = ttk.LabelFrame(main_frame, text="Asetukset ja ohjaus", padding=10)
        control_frame.pack(fill=tk.X, pady=5)

        # Device selection
        device_frame = ttk.Frame(control_frame)
        device_frame.pack(fill=tk.X, pady=5)

        ttk.Label(device_frame, text="Äänilähde:").pack(side=tk.LEFT, padx=5)

        # Get available devices
        devices = self.recorder.get_available_devices()
        device_names = [f"{name} (ID: {id})" for id, name in devices]

        if device_names:
            self.selected_device.set(device_names[0])
            print(f"Oletuslaite asetettu: {device_names[0]}")
        else:
            print("Ei äänilaitteita löytynyt!")

        device_menu = ttk.Combobox(device_frame, textvariable=self.selected_device, values=device_names, width=40, state="readonly")
        device_menu.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Bind selection event
        def on_device_select(_):
            print(f"Äänilähde valittu: {self.selected_device.get()}")

        device_menu.bind("<<ComboboxSelected>>", on_device_select)

        # Language selection
        language_frame = ttk.Frame(control_frame)
        language_frame.pack(fill=tk.X, pady=5)

        ttk.Label(language_frame, text="Kieli:").pack(side=tk.LEFT, padx=5)

        languages = [
            ("Suomi", "fi"),
            ("Englanti", "en"),
            ("Ruotsi", "sv"),
            ("Venäjä", "ru"),
            ("Saksa", "de"),
            ("Ranska", "fr"),
            ("Espanja", "es")
        ]

        language_menu = ttk.Combobox(language_frame, textvariable=self.selected_language,
                                    values=[lang[0] for lang in languages], width=20, state="readonly")
        language_menu.pack(side=tk.LEFT, padx=5)

        # Set default language
        self.selected_language.set("Suomi")
        print(f"Oletuskieli asetettu: Suomi")

        # Speaker diarization checkbox
        diarization_frame = ttk.Frame(control_frame)
        diarization_frame.pack(fill=tk.X, pady=5)

        diarization_check = ttk.Checkbutton(
            diarization_frame,
            text="Käytä puhujan tunnistusta",
            variable=self.use_diarization,
            command=self.toggle_diarization
        )
        diarization_check.pack(side=tk.LEFT, padx=5)

        # Add info label about diarization
        diarization_info = ttk.Label(
            diarization_frame,
            text="(Tunnistaa eri puhujat ja erottelee ne transkriptiossa)",
            foreground='gray'
        )
        diarization_info.pack(side=tk.LEFT, padx=5)

        # Map display language to language code
        def on_language_change(_=None):
            selected = self.selected_language.get()
            for name, code in languages:
                if name == selected:
                    self.transcriber.set_language(code)
                    print(f"Kieli vaihdettu: {name} ({code})")
                    break

        language_menu.bind("<<ComboboxSelected>>", on_language_change)

        # Aseta oletuskieli heti
        on_language_change()

        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Record button
        self.record_button = ttk.Button(button_frame, text="Aloita nauhoitus", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)

        # Clear button
        clear_button = ttk.Button(button_frame, text="Tyhjennä transkriptio", command=self.clear_transcription)
        clear_button.pack(side=tk.LEFT, padx=5)

        # Save button
        save_button = ttk.Button(button_frame, text="Tallenna transkriptio", command=self.save_transcription)
        save_button.pack(side=tk.LEFT, padx=5)

        # Status bar
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=5)

        ttk.Label(status_frame, text="Tila:").pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.status_text).pack(side=tk.LEFT, padx=5)

        # Transcription area (bottom)
        transcription_frame = ttk.LabelFrame(main_frame, text="Transkriptio", padding=10)
        transcription_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Configure text area with better visibility and make it editable
        self.transcription_area = scrolledtext.ScrolledText(
            transcription_frame,
            wrap=tk.WORD,
            font=("TkDefaultFont", 12),
            foreground='black',
            background='white',
            insertbackground='black',  # Cursor color
            undo=True  # Enable undo/redo
        )
        self.transcription_area.pack(fill=tk.BOTH, expand=True)

        # Add a label to indicate that the text is editable
        edit_label = ttk.Label(
            transcription_frame,
            text="Voit muokata transkriptiota suoraan tekstialueella.",
            foreground='blue'
        )
        edit_label.pack(side=tk.BOTTOM, pady=5)

    def toggle_recording(self):
        """Toggle recording on/off."""
        if not self.recording:
            # Start recording
            try:
                # Get the selected device ID
                selected = self.selected_device.get()
                device_id = None

                print(f"Valittu äänilähde: {selected}")

                # Extract the device ID from the selection
                for id, name in self.recorder.get_available_devices():
                    if f"{name} (ID: {id})" == selected:
                        device_id = id
                        print(f"Löydettiin laite ID: {id}, nimi: {name}")
                        break

                if device_id is None:
                    print("Laitetta ei löytynyt, käytetään oletuslaitetta")
                    # Käytä oletuslaitetta, jos valittua laitetta ei löydy
                    device_id = sd.default.device[0]
                    print(f"Oletuslaite: {device_id}")

                # Start the recorder
                print(f"Aloitetaan nauhoitus laitteella: {device_id}")
                self.recorder.start_recording(device_id)

                # Update UI
                self.recording = True
                self.record_button.config(text="Lopeta nauhoitus")
                self.status_text.set("Nauhoitetaan...")
                print("Käyttöliittymä päivitetty: nauhoitetaan")

                # Testaa transkriptiota suoraan
                print("Testataan transkriptiota suoraan...")
                test_text = "Tämä on testitranskriptio. Jos näet tämän tekstin käyttöliittymässä, tekstin päivitys toimii."
                self.ui_update_queue.put(("transcription", test_text))
                self.root.after(1000, self._update_transcription_text, test_text)

            except Exception as e:
                print(f"Virhe nauhoituksen aloittamisessa: {e}")
                messagebox.showerror("Virhe", f"Nauhoituksen aloittaminen epäonnistui: {e}")
        else:
            # Stop recording
            try:
                print("Lopetetaan nauhoitus")
                self.recorder.stop_recording()

                # Update UI
                self.recording = False
                self.record_button.config(text="Aloita nauhoitus")
                self.status_text.set("Nauhoitus pysäytetty")
                print("Käyttöliittymä päivitetty: nauhoitus pysäytetty")

            except Exception as e:
                print(f"Virhe nauhoituksen lopettamisessa: {e}")
                messagebox.showerror("Virhe", f"Nauhoituksen lopettaminen epäonnistui: {e}")

    def on_audio_chunk(self, audio_file):
        """Callback when an audio chunk is recorded."""
        # Add the audio file to the transcription queue
        print(f"Äänipalanen vastaanotettu: {audio_file}")
        self.transcriber.add_audio_file(audio_file)

        # Update status
        status_text = f"Transkriptoidaan... (Jonossa: {self.transcriber.get_queue_size()})"
        print(status_text)
        self.ui_update_queue.put(("status", status_text))

    def on_transcription(self, transcription, _):
        """Callback when transcription is complete."""
        # Add the transcription to the text
        if transcription.strip():
            print(f"Uusi transkriptio vastaanotettu: {transcription[:100]}...")
            print(f"Koko transkriptio: {transcription}")  # Lisätty tulostus

            # Päivitä käyttöliittymä pääsäikeessä
            self.ui_update_queue.put(("transcription", transcription))

            # Päivitä myös suoraan tekstialue (varmuuden vuoksi)
            self.root.after(0, self._update_transcription_text, transcription)
        else:
            print("Tyhjä transkriptio vastaanotettu")

        # Update status
        status_text = f"Transkriptoitu. Jonossa: {self.transcriber.get_queue_size()}"
        print(status_text)
        self.ui_update_queue.put(("status", status_text))

    def _update_transcription_text(self, text):
        """Update transcription text directly."""
        print(f"_update_transcription_text kutsuttu tekstillä: '{text}'")

        if not text.strip():
            print("Teksti on tyhjä, ei päivitetä")
            return

        # Add to the full text
        self.transcription_text += text + "\n\n"

        # Update the text area
        try:
            print(f"Päivitetään tekstialuetta, nykyinen teksti: '{self.transcription_area.get(1.0, tk.END)}'")
            self.transcription_area.delete(1.0, tk.END)
            self.transcription_area.insert(tk.END, self.transcription_text)
            self.transcription_area.see(tk.END)
            print(f"Tekstialue päivitetty suoraan, pituus: {len(self.transcription_text)}")
            print(f"Tekstialueen sisältö päivityksen jälkeen: '{self.transcription_area.get(1.0, tk.END)}'")
        except Exception as e:
            print(f"Virhe tekstialueen suorassa päivityksessä: {e}")

    def update_ui(self):
        """Update the UI from the queue."""
        try:
            # Process all updates in the queue
            while not self.ui_update_queue.empty():
                update_type, data = self.ui_update_queue.get_nowait()

                if update_type == "status":
                    self.status_text.set(data)
                    print(f"Tila päivitetty: {data}")  # Debug tulostus
                elif update_type == "transcription":
                    # Add the transcription to the text area
                    self.transcription_text += data + "\n\n"

                    print(f"Päivitetään tekstialuetta: {data[:100]}...")  # Lisätty tulostus

                    # Tallenna kursorin nykyinen sijainti
                    try:
                        current_cursor_pos = self.transcription_area.index(tk.INSERT)
                        # Tarkista, onko käyttäjä muokkaamassa tekstiä (kursori ei ole lopussa)
                        cursor_at_end = current_cursor_pos == self.transcription_area.index(tk.END + "-1c")

                        # Päivitä tekstialue vain jos kursori on lopussa (käyttäjä ei muokkaa)
                        if cursor_at_end:
                            self.transcription_area.delete(1.0, tk.END)
                            self.transcription_area.insert(tk.END, self.transcription_text)
                            # Scroll to the end
                            self.transcription_area.see(tk.END)
                            print(f"Tekstialue päivitetty, pituus: {len(self.transcription_text)}")  # Debug tulostus
                        else:
                            # Käyttäjä on muokkaamassa tekstiä, joten emme päivitä tekstialuetta
                            print("Käyttäjä muokkaa tekstiä, ei päivitetä tekstialuetta")
                    except Exception as text_error:
                        print(f"Virhe tekstialueen päivityksessä: {text_error}")

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Virhe käyttöliittymän päivityksessä: {e}")

        # Schedule the next update
        self.root.after(100, self.update_ui)

    def clear_transcription(self):
        """Clear the transcription text."""
        self.transcription_text = ""
        self.transcription_area.delete(1.0, tk.END)

    def save_transcription(self):
        """Save the transcription to a file."""
        # Get the current text from the text area (which may have been edited)
        current_text = self.transcription_area.get(1.0, tk.END).strip()

        if not current_text:
            messagebox.showinfo("Tietoa", "Ei transkriptiota tallennettavaksi.")
            return

        try:
            from tkinter import filedialog

            # Create a timestamp for the default filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            default_filename = f"transkriptio_{timestamp}.txt"

            # Ask the user for a filename
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Tekstitiedostot", "*.txt"), ("Kaikki tiedostot", "*.*")],
                initialfile=default_filename,
                title="Tallenna transkriptio"
            )

            if not filename:  # User cancelled
                return

            # Save the transcription
            with open(filename, "w", encoding="utf-8") as f:
                f.write(current_text)

            messagebox.showinfo("Tallennettu", f"Transkriptio tallennettu tiedostoon: {filename}")

        except Exception as e:
            messagebox.showerror("Virhe", f"Tallentaminen epäonnistui: {e}")

    def toggle_diarization(self):
        """Toggle speaker diarization on/off."""
        use_diarization = self.use_diarization.get()
        print(f"Puhujan tunnistus {'käytössä' if use_diarization else 'pois käytöstä'}")

        # Update the transcriber
        self.transcriber.use_diarization = use_diarization

        # Initialize diarization if needed
        if use_diarization and self.transcriber.diarization is None:
            self.transcriber._init_diarization()

    def on_closing(self):
        """Handle window closing."""
        if self.recording:
            self.recorder.stop_recording()

        self.transcriber.stop_processing()
        self.recorder.cleanup()

        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
