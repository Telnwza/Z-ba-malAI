# Azure demo deployment

Deployed on 2026-10-19 (Asia/Bangkok) from commit `db194a905f8cb619c081fc27e96dc70e3f15040b`.

Public website: https://zbamal.malaysiawest.cloudapp.azure.com/

## Resources

- Resource group: `zbamal_group`
- VM: `zbamal`, Ubuntu 24.04 LTS x64, Trusted Launch
- Region: Malaysia West, zone 1
- Size: `Standard_B2as_v2`, 2 vCPU, 8 GiB RAM
- OS disk: 64 GiB Standard SSD LRS (E6)
- Public IP resource: `zbamal-ip`
- Public inbound ports: HTTP 80 and HTTPS 443 only; public SSH 22 is blocked
- No load balancer, paid backup, or paid monitoring extension was added

## Cost and lifetime

Microsoft retail prices checked at deployment:

| Component | USD |
| --- | ---: |
| VM | 0.0855/hour |
| Standard IPv4 public IP | 0.005/hour |
| E6 LRS disk | 4.80/month |
| E6 LRS disk mount | 0.55/month |

Estimated fixed total: **70.51 USD for 30 days**, or **72.68 USD for 31 days**.
Disk operations, boot-diagnostic storage, chargeable outbound data transfer, and applicable taxes are additional. The remaining allowance below 100 USD is a buffer for normal demo usage; it is not a guaranteed resource-level spending cap.

Budget `zbamal-100usd-demo` is set at 100, resets monthly, expires on 2026-11-30, and emails the account owner at 75% and 90% actual cost. A budget sends alerts; it does not stop the VM.

Automatic stop is **Enabled** as described below. For manual shutdown, open VM `zbamal` in Azure Portal, use **Stop**, and verify **Stopped (deallocated)**. Disks and the public IP can continue to incur charges while retained. Do not rely on guest `shutdown` alone to stop compute billing.

## Automatic stop

Enabled on 2026-10-19 after the owner confirmed the permissions and activation.

- Consumption Logic App: `zbamal-autostop`, resource group `zbamal_group`, Japan East.
- Deadline: **2026-11-19 00:00 Asia/Bangkok**, equivalent to **2026-11-18 17:00 UTC**.
- Cost threshold: **80 USD** reported cumulative subscription spending, using budget `zbamal-80usd-cumulative`.
- The cost budget is quarterly, starts on 2026-10-01, and expires on 2026-11-30. It does not reset on November 1.
- The workflow checks every six hours, at approximately 00:00, 06:00, 12:00, and 18:00 in Thailand. The first scheduled check is 2026-10-19 06:00.
- At each check, it evaluates the deadline first. Otherwise it reads the cumulative budget and deallocates the VM if the reported currency is USD and the amount is at least 80. Billing-read failures cannot block the deadline branch.
- Granted permissions: custom role `Zbamal autostop - budget read` has only `Microsoft.Consumption/budgets/read` at subscription scope; custom role `Zbamal autostop - deallocate VM` has only `Microsoft.Compute/virtualMachines/deallocate/action` at the `zbamal` VM scope. No VM deletion, start, source modification, or general Contributor permission is needed.
- The workflow runs in Azure and does not depend on this computer being on. It remains scheduled after the deadline; disable `zbamal-autostop` before intentionally restarting the VM after a limit has been reached.

The spending feed currently reports `0.0` with an empty currency. This is not proof of zero actual usage. The workflow conditions passed three Azure-native mock tests (below both limits, exactly 80 USD, and past the deadline), but stopping on a real billed threshold cannot yet be verified. Cost reporting and polling delays can allow the amount to exceed 80 USD before shutdown; this is not a hard spending cap.

Logic Apps Consumption bills per execution/action and may add a small usage charge. Retained disk and public IP charges continue after VM deallocation. No Azure Automation account was created: the Students offer's Automation region restriction conflicted with this subscription's allowed regions.

## Runtime layout

- Application snapshot: `/opt/zbamal`
- Python environment: `/opt/zbamal/.venv`
- CPU-only PyTorch and the repository requirements were installed
- Model/cache: `/var/lib/zbamal/huggingface`
- API service: `zbamal.service`, one Uvicorn worker, bound to `127.0.0.1:8000`
- Frontend: `/opt/zbamal/src/frontend`, served by Nginx
- Nginx site: `/etc/nginx/sites-available/zbamal`
- Install log: `/var/log/zbamal-install.log`
- HTTPS: Let's Encrypt with Certbot/Nginx; HTTP redirects to HTTPS

The `demo` source and deployed `app.js` use the same origin on the public website. Local static previews on localhost or file URLs keep the `localhost:8000` fallback, and `?api=...` remains available. The Nginx proxy limits request bodies to 16 KiB and translation requests to 5 requests/second per IP with a burst allowance of 10.

Use **VM > Operations > Run command > RunShellScript** for administration without exposing SSH. Useful commands:

```sh
systemctl status zbamal nginx --no-pager
journalctl -u zbamal -n 50 --no-pager
systemctl restart zbamal
systemctl status certbot.timer --no-pager
```

Keep the downloaded SSH private key outside the repository. It is not included in these deployment notes or application files.

## Verified

- Repository tests: 10 passed locally and 10 passed on the Ubuntu VM. Similarity scores are mocked in these tests.
- Real model: both `/translate` directions returned HTTP 200 over verified HTTPS.
- Empty text returned HTTP 422.
- HTTP redirected to HTTPS.
- Public SSH 22 was not reachable from the test machine.
- Browser: both translation directions worked; selecting an alternate candidate updated the output.
- Auto-stop: workflow and trigger are Enabled; the next scheduled check is 2026-10-19 06:00 Asia/Bangkok. A live run using the managed identity read the cumulative budget successfully, skipped both deallocation actions, and kept the VM running. HTTPS still returned 200 afterward.
- Auto-stop conditions: three Azure-native mock runs passed for 79.99 USD before the deadline, exactly 80 USD, and a passed deadline. Production has no static mock results. An actual VM deallocation was not performed during verification.

These checks establish a working demo, not a general accuracy score or a long-duration availability test.

## Dataset refresh on 2026-10-19

- Source: Gemini artifact `thai_chat_genz_dataset_v13.xlsx`, copied unchanged to `data/dataset.xlsx` on local branch `demo`.
- Workbook SHA-256: `b16cba923144d2a31a64c379ae762631c1a1baa211b85b558d022cdf84ba3aa7`.
- Active rules: 234 (previously 154). Training examples: 1,549. Contrast examples: 552.
- Matching `E_train.npy`, `E_contrast.npy` and `metadata.json` were rebuilt with the existing `BAAI/bge-m3` builder, using 1,024-dimensional normalized embeddings. Existing matching sentence vectors were reused.
- The workbook has cached formula values, no cached Excel errors, no duplicate active Rule IDs and no missing source, target or training fields.
- VM backup before the refresh: `/var/backups/zbamal/dataset-before-demo-v13-20261019/`. Staging and rebuild files: `/var/lib/zbamal/demo-v13-20261019/`.
- The VM build validated the workbook/metadata/vector correspondence before replacing the four files and restarting `zbamal.service`. It retained a rollback path for build or test failures. The build completed successfully and the service is active.
- Verification: 10 repository tests passed locally and 10 on the VM. Six additional real-model smoke cases passed locally and over public HTTPS, including the new terms `เก็ต`, `ช็อต` and `ตปท`, both directions, response spans and empty-input HTTP 422. These examples are smoke checks, not an accuracy benchmark.
- The frontend keeps the existing GitHub link and styling. No public ZIP export was created.
- Demo changes are maintained on branch `demo`.
