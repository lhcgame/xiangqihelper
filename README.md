# Pikafish Screen Helper

A small Python helper for Xiangqi. It watches your screen, reads the board with
calibrated piece templates, sends the position to Pikafish, and prints the best
move. It can also click the move for you if auto move is enabled.

## What You Need

- Python 3
- Pikafish and `pikafish.nnue`
- A Xiangqi board visible on your primary monitor

The app looks for engines in this order:

- `engines/Windows/` on Windows
- `engines/Linux/` on Linux
- `engines/MacOS/` on macOS
- `./pikafish.exe` on Windows, or `./pikafish` on Linux/macOS

It picks the best engine binary for the client CPU when several Pikafish builds
are in the platform folder.

On macOS, give your terminal app Screen Recording permission. If you use auto
move, also give it Accessibility permission.

On Linux, use an X11 session for auto move. If mouse control is not available,
set `"auto_move": false` and the helper can still print moves.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Calibrate

Calibrate once for the board style you are using.

1. Open your Xiangqi board.
2. Put the board in the standard starting position.
3. Run:

```bash
python3 calibrate.py
```

4. Click the top-left grid intersection.
5. Click the bottom-right grid intersection.
6. Press Enter when the green grid preview lines up.

Calibration saves:

- `config.json`
- `templates/ref_board.png`
- one template image for each piece in `templates/`

Recalibrate if you change websites, board themes, zoom level, or window size.

For a second monitor, set `"screen_monitor": 2` in `config.json` before
calibrating. Use `0` to capture the whole desktop.

## Run

```bash
python3 main.py
```

This opens a small control window.

- Start / Stop: begin or stop watching the board
- Reload: reload calibration/settings and recalculate
- Calibrate: stop the helper and open calibration
- Config: set timing, recognize the next-game button, and enable auto next game
- Exit: stop the helper and close Pikafish

If Start says `config.json missing`, click Calibrate first. The calibration
window creates `config.json` for that app folder.

The helper now uses the turn avatar as its trigger. When your green turn ring is
active, it reads the current board, generates FEN, asks Pikafish for the best
move, clicks that move, then waits until your next turn.

For the old terminal-only mode, run:

```bash
python3 main.py --cli
```

## Auto Move

Auto move is controlled in `config.json`.

The GUI can save the common timing, turn, and next-game settings:

- Move time: search for a random time from the first number to the second number
- Recognize Turn: select the rectangle around your avatar green countdown ring.
- Use turn avatar: when that green ring is active on a stable board, search and
  auto move once, then wait for the ring to go away before moving again.
- Recognize Next: on the game-over screen, click the center of the Next Game
  button and press Enter. This saves `templates/next_game_button.png`.
- Auto start next game: when the saved button is visible, click it once and wait
  for the new starting position.
- Active next screen scan: while running, scan the current screen for the saved
  next-game button and click it as soon as it appears.

To only print moves and never click:

```json
"auto_move": false
```

To let the helper click moves:

```json
"auto_move": true
```

If clicks land too high or too low, adjust:

```json
"auto_move_click_offset_y": -10
```

Negative values click higher. Positive values click lower.

## Files

- `main.py`: runs the helper
- `calibrate.py`: creates board and piece templates
- `vision.py`: reads the board from the screen
- `engine.py`: starts Pikafish and asks for moves
- `app_paths.py`: finds files in source and packaged builds
- `xiangqi_core.py`: validates board states and detects moves
- `config.json`: calibration and auto-move settings
- `templates/`: saved board and piece images

## Troubleshooting

If calibration is bad, run `python3 calibrate.py` again and click the grid
intersections, not the piece centers.

If the engine does not start on Linux/macOS, make sure `pikafish` is executable
and the NNUE file exists:

```bash
ls -l pikafish pikafish.nnue
```

On Windows, use `pikafish.exe` or put the downloaded `.exe` files in
`engines/Windows/`.

If the helper cannot see or click the board on macOS, check Screen Recording and
Accessibility permissions for your terminal app.
