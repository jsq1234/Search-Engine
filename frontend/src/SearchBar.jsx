
export default function SearchBar({ queryVar , setQuery, onSubmit}) {
    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit();
    }

    return (
        <div className='w-[60%] mx-auto mt-10 text-xl'>
            <form onSubmit={handleSubmit}>
                <div className='flex'>
                    <input
                        type="text"
                        id="query"
                        name="query"
                        value={queryVar}
                        onChange={(e) => setQuery(e.target.value)}
                        required
                        className='text-xl flex-1 rounded-lg rounded-r-none focus:outline-0'
                    />
                    <button type="submit" className='p-3 hover:bg-slate-900 rounded-lg rounded-l-none bg-black dark:text-blue-100'>
                        Search
                    </button>
                </div>
            </form>
        </div>
    )
}
