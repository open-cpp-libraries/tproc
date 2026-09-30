// SPDX-License-Identifier: BSL-1.0
// Copyright 2025-2026, Amlal El Mahrouss (amlal@nekernel.org)
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// Official repository: https://github.com/open-cpp-libraries/tproc

#ifndef OCL_TPROC_ROPE_HPP
#define OCL_TPROC_ROPE_HPP

#include <ocl/tproc/detail/rope_fwd.hpp>

namespace ocl::tproc::rope
{

	/// \brief reverse exact pred type.
	template <typename It>
	class reverse_pred final
	{
		It cond_;

	public:
		reverse_pred(const boost::core::basic_string_view<typename It::value_type>& cond)
			: cond_(cond)
		{
		}

		using iterator_ptr = It*;

		iterator_ptr operator()(iterator_ptr rbegin, iterator_ptr rend)
		{
			for (auto rbeg{rbegin}; rbeg != rend; ++rbeg)
			{
				if (*rbeg == cond_)
					return rbeg;
			}

			return rend;
		}
	};

	/// \brief ends with pred type.
	template <typename It>
	class ends_with_pred final
	{
		It cond_;

	public:
		ends_with_pred(const boost::core::basic_string_view<typename It::value_type>& cond)
			: cond_(cond)
		{
		}

		using iterator_ptr = It*;

		iterator_ptr operator()(iterator_ptr begin, iterator_ptr end)
		{
			for (auto beg{begin}; beg != end; ++beg)
			{
				if (beg->ends_with(cond_))
					return beg;
			}

			return end;
		}
	};

	template <typename It>
	class uppercase_pred final
	{
		It cond_;

	public:
		uppercase_pred(const boost::core::basic_string_view<typename It::value_type>& cond)
			: cond_(cond)
		{
		}

		using iterator_ptr = It*;

		iterator_ptr operator()(iterator_ptr begin, iterator_ptr end)
		{
			std::transform(cond_.begin(),
						   cond_.end(),
						   [](std::allocator_traits<It>::value_type& ch) {
							   return std::toupper(ch);
						   });

			for (auto beg{begin}; beg != end; ++beg)
			{
				if (*beg == cond_)
					return beg;
			}

			return end;
		}
	};

	template <typename It>
	class lowercase_pred final
	{
		It cond_;

	public:
		lowercase_pred(const boost::core::basic_string_view<typename It::value_type>& cond)
			: cond_(cond)
		{
		}

		using iterator_ptr = It*;

		iterator_ptr operator()(iterator_ptr begin, iterator_ptr end)
		{
			std::transform(cond_.begin(),
						   cond_.end(),
						   [](std::allocator_traits<It>::value_type& ch) {
							   return std::tolower(ch);
						   });

			for (auto beg{begin}; beg != end; ++beg)
			{
				if (*beg == cond_)
					return beg;
			}

			return end;
		}
	};

	template <typename It>
	class exact_pred final
	{
		It cond_;

	public:
		exact_pred(const boost::core::basic_string_view<typename It::value_type>& cond)
			: cond_(cond)
		{
		}

		using iterator_ptr = It*;

		iterator_ptr operator()(iterator_ptr begin, iterator_ptr end)
		{
			for (auto beg{begin}; beg != end; ++beg)
			{
				if (*beg == cond_)
					return beg;
			}

			return end;
		}
	};

	template <typename It>
	class starts_with_pred final
	{
		It cond_;

	public:
		starts_with_pred(const boost::core::basic_string_view<typename It::value_type>& cond)
			: cond_(cond)
		{
		}

		using iterator_ptr = It*;

		iterator_ptr operator()(iterator_ptr begin, iterator_ptr end)
		{
			for (auto beg{begin}; beg != end; ++beg)
			{
				if (beg->starts_with(cond_))
					return beg;
			}

			return end;
		}
	};

} // namespace ocl::tproc::rope

namespace ocl::tproc
{

	template <typename It, class Pred>
	Pred find(It begin, It end, Pred);

	template <typename It, class Pred>
	Pred find_if(It begin, It end, Pred);

	template <typename It, class Pred>
	Pred::size_type erase(It begin, It end, Pred);

	template <typename It, class Pred>
	Pred::size_type erase_if(It begin, It end, Pred);

} // namespace ocl::tproc

inline std::ostream& operator<<(std::ostream& os, ocl::tproc::crope& r)
{
	os << r.to_string();
	return os;
}

inline std::wostream& operator<<(std::wostream& os, ocl::tproc::wrope& r)
{
	os << r.to_string();
	return os;
}

inline std::ostream& operator<<(std::ostream& os, const ocl::tproc::crope& r)
{
	os << r.to_string();
	return os;
}

inline std::wostream& operator<<(std::wostream& os, const ocl::tproc::wrope& r)
{
	os << r.to_string();
	return os;
}

#include "rope.inl"

#endif // OCL_TPROC_ROPE_HPP
