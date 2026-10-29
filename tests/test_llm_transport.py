"""Interrupted HTTP responses retry without changing prompts or losing completed caches."""
from pathlib import Path
import http.client
import io
import json
import sys
import tempfile
import urllib.error
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umsm.llm import LLMClient


def response(text='ok'):
    return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': text}}],
                                'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}).encode())


class BrokenBody(io.BytesIO):
    def read(self, *args):
        raise http.client.IncompleteRead(b'{"partial":', 2)


def test_incomplete_and_invalid_responses_retry_identical_requests():
    for failure in (BrokenBody(), http.client.RemoteDisconnected('connection closed'),
                    ConnectionResetError('connection reset'), TimeoutError('timed out'),
                    io.BytesIO(b'{"unfinished":'), io.BytesIO(b'\xff')):
        client = LLMClient(max_retries=2)
        with patch('umsm.llm.read_api_key', return_value='test-key'), \
                patch('urllib.request.urlopen', side_effect=[failure, response()]) as request, \
                patch('umsm.llm.time.sleep') as sleep:
            assert client.chat([{'role': 'user', 'content': 'same prompt'}]) == 'ok'
            assert request.call_count == 2
            first, second = [call.args[0] for call in request.call_args_list]
            assert first.data == second.data and first.full_url == second.full_url
            sleep.assert_called_once_with(2.)
        assert client.n_calls == 1 and client.prompt_tokens == 10 and client.completion_tokens == 2


def test_transport_exhaustion_is_bounded_and_redacted():
    secret = 'sk-unit-test-only'
    client = LLMClient(max_retries=2)
    with patch('umsm.llm.read_api_key', return_value=secret), \
            patch('urllib.request.urlopen', side_effect=urllib.error.URLError(secret)) as request, \
            patch('umsm.llm.time.sleep') as sleep:
        try:
            client.chat([])
        except RuntimeError as exc:
            assert '3 attempt(s)' in str(exc) and 'cached' in str(exc)
            assert secret not in str(exc) and '[REDACTED]' in str(exc)
        else:
            raise AssertionError('failed responses must not become matcher scores')
        assert request.call_count == 3 and [c.args[0] for c in sleep.call_args_list] == [2., 4.]
    assert client.n_calls == 0 and not client._memory


def test_fatal_status_with_broken_body_does_not_retry():
    for status in (401, 402):
        failure = urllib.error.HTTPError('https://example.invalid', status, 'fatal', {}, BrokenBody())
        with patch('umsm.llm.read_api_key', return_value='test-key'), \
                patch('urllib.request.urlopen', side_effect=failure) as request, \
                patch('umsm.llm.time.sleep') as sleep:
            try:
                LLMClient(max_retries=5)._request([])
            except RuntimeError as exc:
                assert ('invalid API key' if status == 401 else 'insufficient balance') in str(exc)
            else:
                raise AssertionError('fatal status must fail')
            assert request.call_count == 1 and sleep.call_count == 0


def test_prefetch_restart_reuses_completed_disk_responses():
    with tempfile.TemporaryDirectory() as folder:
        jobs = [('group', [{'role': 'user', 'content': name}]) for name in ('saved', 'interrupted')]
        calls = []
        def interrupted(request, **kwargs):
            name = json.loads(request.data)['messages'][0]['content']
            calls.append(name)
            return response(name) if name == 'saved' else BrokenBody()
        client = LLMClient(cache_dir=folder, max_retries=0)
        keys = [client._key(messages, 0) for _, messages in jobs]
        with patch('umsm.llm.read_api_key', return_value='test-key'), \
                patch('urllib.request.urlopen', side_effect=interrupted):
            try:
                client.prefetch(jobs, workers=1, settle_s=0)
            except RuntimeError:
                pass
            else:
                raise AssertionError('interrupt must propagate after its retry budget')
        assert Path(client._cache_path(keys[0])).is_file()
        assert not Path(client._cache_path(keys[1])).exists()
        restarted = LLMClient(cache_dir=folder, max_retries=0)
        assert [restarted._key(messages, 0) for _, messages in jobs] == keys
        with patch('umsm.llm.read_api_key', return_value='test-key'), \
                patch('urllib.request.urlopen', return_value=response('interrupted')) as request:
            restarted.prefetch(jobs, workers=1, settle_s=0)
            assert request.call_count == 1
            assert json.loads(request.call_args.args[0].data)['messages'][0]['content'] == 'interrupted'
            assert restarted.chat(jobs[0][1]) == 'saved'
        assert not list(Path(folder).rglob('*.tmp'))
        assert len(list(Path(folder).rglob('*.json'))) == 2


def test_cache_write_failure_leaves_no_partial_json():
    with tempfile.TemporaryDirectory() as folder:
        client = LLMClient(cache_dir=folder)
        with patch.object(client, '_request', return_value='ok'), \
                patch('umsm.llm.os.replace', side_effect=OSError('disk unavailable')):
            try:
                client.chat([])
            except OSError:
                pass
            else:
                raise AssertionError('cache publication failure must remain visible')
        assert not list(Path(folder).rglob('*.json')) and not list(Path(folder).rglob('*.tmp'))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
