from rq.job import Job
from redis import Redis

job_id = 'ed610eb1-810e-4301-bd0a-08a53cf5d415'
try:
    r = Redis.from_url('redis://redis:6379/0')
    j = Job.fetch(job_id, connection=r)
    print('status:', j.get_status())
    print('started_at:', j.started_at)
    print('ended_at:', j.ended_at)
    print('exc_info:', bool(j.exc_info))
    if j.exc_info:
        print('exc_info_text:', j.exc_info)
    print('worker_name:', getattr(j, 'origin', None))
    print('meta keys:', list(j.meta.keys()))
except Exception as e:
    print('Error fetching job:', e)
