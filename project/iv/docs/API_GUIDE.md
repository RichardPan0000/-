# IV Diagnosis Service API Guide

Base URL: `http://<host>:<port>`

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check and task-manager snapshot |
| `POST` | `/diagnosis/tasks` | Submit an async diagnosis task |
| `POST` | `/diagnosis/tasks/query` | Query async task status by `sn_code + task_id` |
| `POST` | `/diagnosis` | Run synchronous diagnosis |

## 1. Submit Async Diagnosis Task

```http
POST /diagnosis/tasks
Content-Type: application/json
```

Request body:

```json
{
  "sn_code": "120A13AT0220",
  "task_id": 26321
}
```

Response example:

```json
{
  "success": true,
  "message": "task queued",
  "data": {
    "sn_code": "120A13AT0220",
    "task_id": 26321,
    "status": "queued",
    "created_at": 1746612000.123,
    "started_at": null
  }
}
```

Notes:

- Public uniqueness key is `sn_code + task_id`
- Re-submitting the same task does not enqueue it twice
- Duplicate submit returns `message: "task already exists"`

## 2. Query Async Task Status

```http
POST /diagnosis/tasks/query
Content-Type: application/json
```

Request body:

```json
{
  "sn_code": "120A13AT0220",
  "task_id": 26321
}
```

Response example:

```json
{
  "success": true,
  "message": "task found",
  "data": {
    "sn_code": "120A13AT0220",
    "task_id": 26321,
    "status": "queued",
    "created_at": 1746612000.123,
    "started_at": 1746612001.456
  }
}
```

Not found response:

```json
{
  "detail": "Diagnosis task not found"
}
```

HTTP status code is `404`.

## 3. Public Status Model

Only these public statuses are exposed:

| Status | Meaning |
|---|---|
| `queued` | Task is queued or currently running |
| `success` | Task finished successfully |
| `fail` | Task execution failed, or backend callback failed |

Internal status mapping:

- internal `running` -> public `queued`
- internal `failed` / `callback_failed` -> public `fail`

## 4. Typical Async Flow

```text
1. POST /diagnosis/tasks
2. Poll POST /diagnosis/tasks/query
3. Stop when status becomes success or fail
```

Recommended polling interval: 2 to 5 seconds.

## 5. Backend Callback

This callback is for backend-to-backend integration only. Frontend should not call it directly.

### Callback configuration

The diagnosis service reads callback settings from the existing `api_service.callback.*` config:

```yaml
api_service:
  callback:
    enabled: true
    url: "http://10.1.51.93:30909/api/iv/algo/callback"
    token: ""
    retries: 3
    timeout_seconds: 10
```

Notes:

- `url` should be the full callback URL
- callback is disabled when `enabled=false` or `url` is empty
- current integration does not send an auth header

### Callback request

The diagnosis service sends callback only on terminal states:

- `SUCCESS`
- `FAILED`

It does not send `QUEUED`.

```http
POST /api/iv/algo/callback
Content-Type: application/json
```

Success payload example:

```json
{
  "snCode": "120A13AT0220",
  "taskId": "26321",
  "status": "SUCCESS",
  "finishedDateTime": "2026-05-08 16:30:15"
}
```

Failed payload example:

```json
{
  "snCode": "120A13AT0220",
  "taskId": "26321",
  "status": "FAILED",
  "finishedDateTime": "2026-05-08 16:31:02",
  "msg": "diagnosis failed"
}
```

Callback success is defined as:

- HTTP success response
- JSON response body with `code == 0`

Business error responses such as `code != 0` are treated as callback failure.

### Callback mechanism

Internal task state flows like this:

```text
queued -> running -> success/failed -> callback
```

Mechanism details:

- When a task is submitted, its internal state starts as `queued`
- When a worker actually begins execution, the internal state becomes `running`
- If diagnosis completes normally, the internal state becomes `success`
- If diagnosis raises an exception, the internal state becomes `failed`
- Callback is triggered only after the task has already entered `success` or `failed`
- Before sending callback, the service records `finished_at` and uses it to build `finishedDateTime`

Important notes:

- `queued` and `running` do not trigger callback
- The public query API still exposes only `queued / success / fail`
- Internal `running` is projected to public `queued`
- If task execution failed, callback payload uses `status=FAILED`

If backend callback itself fails after all retries:

- internal state becomes `callback_failed`
- public query status is still exposed as `fail`
- the original diagnosis result is not re-run automatically; the failure is in callback delivery, not diagnosis execution

## 6. Load-Test Script

The repository provides:

```text
scripts/load_test_diagnosis_api.py
```

The script exercises:

1. `POST /diagnosis/tasks`
2. `POST /diagnosis/tasks/query`
3. `GET /health`

It no longer uses:

- `submission_id`
- `GET /diagnosis/tasks/submissions/{submission_id}`
- `GET /diagnosis/tasks/{station_id}/{sn_code}/{task_id}`

Example:

```powershell
D:\software\miniforge\install\envs\python310\python.exe -X utf8 scripts\load_test_diagnosis_api.py --base-url http://127.0.0.1:3602 --sn-code 120A13AT0220 --task-id 26321 --requests 5 --submit-workers 5
```

For load testing, it is recommended to disable callback unless you are explicitly testing callback integration.

## 7. Health Check

```http
GET /health
```

Minimal response example:

```json
{
  "status": "ok"
}
```

Depending on runtime configuration, the response may also include `task_manager` and `resources`.

## 8. Synchronous Diagnosis

```http
POST /diagnosis
Content-Type: application/json
```

This endpoint blocks until diagnosis is finished. It is useful for debugging, but not recommended for production task orchestration.

## 9. Removed Legacy Endpoints

These endpoints are no longer provided:

- `GET /diagnosis/tasks/{station_id}/{sn_code}/{task_id}`
- `GET /diagnosis/tasks/submissions/{submission_id}`
