from libextract import build_parser, run_extract

if __name__ == "__main__":
    args = build_parser().parse_args()
    run_extract(
        args.url,
        args.storage_mode,
        args.output,
        args.db_uri,
        args.verbose
    )
