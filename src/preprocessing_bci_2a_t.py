import mne
import numpy as np

def preprocess_bci_2a_t(file_path):

    # Load GDF file
    # file_path = "/home/gs/code/AS24-V59/project/data/BCICIV_2a_gdf/A02T.gdf"  # Replace with the actual file path
    raw = mne.io.read_raw_gdf(file_path, preload=True)

    # Pick channels 1-22
    raw.pick(range(22))

    print("----------------------------------------------------")
    print(f"Channel names: {raw.ch_names}")
    print("----------------------------------------------------")

    eeg_found = True
    for c in raw.ch_names:
        if "EEG" not in c:
            print(f"Warning: Channel {c} does not contain 'EEG'")
            eeg_found = False
            break

    if eeg_found:
        print("----------------------------------------------------")
        print(f"All {raw.filenames}'s channels contain 'EEG'")
        print("----------------------------------------------------")

    # Extract events (stimuli codes)
    events, event_ids = mne.events_from_annotations(raw)

    # Print event information
    # print("Events:", events)
    # print("Event IDs:", event_ids)

    # Plot events on the raw data
    # mne.viz.plot_events(events, raw.info["sfreq"], )

    # Apply band-pass filter
    raw.filter(4, 40, method="iir")

    eeg_data = raw.get_data()

    freq = int(raw.info["sfreq"])

    # Define epoch duration (0-4 seconds after cue onset)
    tmin, tmax = 0 * freq, 4 * freq  # Seconds relative to event onset

    X = np.empty((0, 22, 1000))
    y = np.empty(0)

    # Event ID 768: Start of a trial
    # Event ID 769: Cue onset left (class 1)
    # Event ID 770: Cue onset right (class 2)
    # Event ID 771: Cue onset foot (class 3)
    # Event ID 772: Cue onset tongue (class 4)
    # Event ID 783: Cue onset unknown

    event_interest = [event_ids["769"], event_ids["770"], event_ids["771"], event_ids["772"]]

    reverse_event_ids = {v: k for k, v in event_ids.items()}

    for event in events:
        # print(event)
        if (event[2] in event_interest):

            x_cut = eeg_data[:, event[0] + tmin : event[0] + tmax]
            X = np.concatenate((X, x_cut[np.newaxis, :]), axis=0)
            y = np.append(y, reverse_event_ids[event[2]])

            # print(event)

    # Z-score normalization
    X = (X - np.mean(X, axis=(0, 2), keepdims=True)) / np.std(X, axis=(0, 2), keepdims=True)

    X = X[:, np.newaxis, :, :]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y
