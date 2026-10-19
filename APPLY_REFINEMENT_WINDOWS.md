# SquatVision — apply the review refinement on Windows

The patch was generated from **the source ZIP you supplied in this conversation**,
not from GitHub. It changes the current project incrementally.

From the root of your local SquatVision repository:

```powershell
$patch = Join-Path $HOME 'Downloads\SquatVision_review_refinement.patch'

git -c core.autocrlf=true apply --check $patch
if ($LASTEXITCODE -ne 0) { throw 'Patch mismatch: stop; do not force changes.' }

git -c core.autocrlf=true apply $patch
if ($LASTEXITCODE -ne 0) { throw 'Patch could not be applied.' }

python -m unittest discover -s tests -v

python app.py --input data/squat.mp4 --output outputs/review_refinement --side right --start-time 64 --end-time 80 --min-visibility 0.30

$report = Get-Content outputs/review_refinement/summary.json -Raw | ConvertFrom-Json
$report.auto_bottom_reference_frames | Select-Object rep_number,frame,status,reason,valid_proposal_frames,suggested_review_frame | Format-List

python manual_depth.py --input data/squat.mp4 --analysis-summary outputs/review_refinement/summary.json --rep-number 1 --open-best-candidate --output data/squat_manual.json
```

Manual review opens a GUI. Its suggested frame is **not automatically verified**;
you must visually check both points and reposition if necessary before saving.

If you have uncommitted source edits since exporting the ZIP, check their
compatibility first and preserve a separate backup. The full source ZIP is a
fallback copy; **do not extract it over your project without checking the
changes**. Exclude `.venv`, `.git`, videos and raw outputs from Git commits.

The patch improves review and a constrained knee silhouette contour; it does
not yet solve hidden upper-thigh anatomy. New tests are synthetic/unit tests,
not numerical evidence of IPF depth accuracy on real footage.
