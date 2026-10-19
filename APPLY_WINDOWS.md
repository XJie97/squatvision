# Apply the surface-review update

Run from the existing SquatVision repository. Preserve any uncommitted work first.
The per-command `core.autocrlf=true` setting handles the supplied Windows CRLF
source files without changing your repository configuration. The patch is based on the supplied source-only ZIP, including its automatic-bottom
update; it is not based on a generic v5 release.

```powershell
cd "C:\Users\lohxi\OneDrive\Desktop\squatvision_v2_documented\squatvision"
git status
$patch = Join-Path $HOME "Downloads\SquatVision_surface_review.patch"
git -c core.autocrlf=true apply --check $patch
if ($LASTEXITCODE -ne 0) { throw "Patch does not match this checkout; no changes applied." }
git -c core.autocrlf=true apply $patch
if ($LASTEXITCODE -ne 0) { throw "Patch application failed." }
python -m unittest discover -s tests -v
```

If only README/CHANGELOG conflict, use `SquatVision_surface_review_code_only.patch`
with the same check-then-apply sequence. Do not apply both patches. The code-only
patch includes all Python changes/tests; reconciled docs and dependency files are
in the full source ZIP. If Python files conflict, extract the source ZIP into a
separate directory and compare with the local checkout; do not force-apply.

For the source ZIP, extract to a new directory rather than over an existing
repository. Add your original videos to its data folder, create a Python 3.12
virtual environment, install `requirements-tested.txt`, and run
`python download_model.py`. Videos, model weights and generated outputs are not
bundled in the source ZIP.

```powershell
python app.py
python manual_depth.py --input data/squat.mp4 --analysis-summary outputs/squat/summary.json --rep-number 1 --output data/annotations/squat.json
python app.py --annotations-dir data/annotations
```

Review opens at the detected reference without entering a timestamp. U explicitly
accepts a visible pair; click hip then knee to replace; A/D navigates; S saves;
Esc/window X cancels. Automatic recognition on the supplied recording remains
unresolved; the validation report documents rejected proposals and evidence.
