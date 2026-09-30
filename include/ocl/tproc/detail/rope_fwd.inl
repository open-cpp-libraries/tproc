// SPDX-License-Identifier: BSL-1.0
// Copyright 2025-2026, Amlal El Mahrouss (amlal@nekernel.org)
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// Official repository: https://github.com/open-cpp-libraries/tproc

#ifndef OCL_TPROC_ROPE_FWD_INL
#define OCL_TPROC_ROPE_FWD_INL

#include <boost/system/error_code.hpp>
#include <iostream>
#include <stdexcept>

namespace ocl::tproc
{

	/// \brief PImpl of the rope algorithm.
	template <class CharT, class Traits, class Allocator>
	struct basic_rope<CharT, Traits, Allocator>::tree_impl
	{
		using char_type		 = CharT;
		using value_type	 = char_type;
		using size_type		 = typename std::allocator_traits<Allocator>::size_type;
		using error_type	 = boost::system::error_code;
		using allocator_type = Allocator;
		using rope_ptr		 = basic_rope<CharT, Traits, Allocator>::rope_ptr;

		// B-Tree fields.
		rope_ptr left_{nullptr};  // Left child (internal node only)
		rope_ptr right_{nullptr}; // Right child (internal node only)

		size_type	   weight_{0};	   // Size of left subtree (internal) OR to_string size (leaf)
		value_type*	   blob_{nullptr}; // Character to_string (leaf node only)
		allocator_type alloc_;
		size_type	   capacity_{0}; // Allocated blob capacity

	public:
		tree_impl(Allocator alloc = Allocator())
			: alloc_(alloc)
		{
		}

		bool is_leaf() const
		{
			return blob_ != nullptr;
		}

		tree_impl(const boost::core::basic_string_view<CharT>& str,
				  Allocator									   alloc = Allocator())
			: weight_(str.size()), alloc_(alloc), capacity_(str.size())
		{
			if (weight_ > 0)
			{
				blob_ = alloc_.allocate(capacity_);

				if (!blob_)
					detail::throw_bad_alloc();

				::memset(blob_, 0, capacity_);
				Traits::copy(blob_, str.data(), weight_);
			}
		}

		tree_impl(rope_ptr	left,
				  rope_ptr	right,
				  Allocator alloc = Allocator())
			: left_(left), right_(right), alloc_(alloc)
		{
			weight_ = left ? left->impl_->total_size() : 0;
		}

		~tree_impl()
		{
			if (blob_ && capacity_ > 0)
			{
				alloc_.deallocate(blob_, capacity_);
			}
		}

		size_type total_size() const
		{
			if (is_leaf())
			{
				return weight_;
			}

			return weight_ + (right_ ? right_->impl_->total_size() : 0);
		}

		size_type size() const
		{
			return total_size();
		}

		CharT at(size_type pos) const
		{
			if (is_leaf())
			{
				if (pos >= weight_)
				{
					throw std::out_of_range("rope index out of range");
				}

				return blob_[pos];
			}

			// Internal node: navigate based on weight
			if (pos < weight_)
			{
				if (!left_)
					throw std::out_of_range("null left child");
				return left_->impl_->at(pos);
			}
			else
			{
				if (!right_)
					throw std::out_of_range("null right child");
				return right_->impl_->at(pos - weight_);
			}
		}

	public:
		rope_ptr concat(rope_ptr left,
						rope_ptr right)
		{
			if (!left)
				return right;

			if (!right)
				return left;

			right->impl_->left_ = left;
			left->impl_->right_ = right;

			return left;
		}

		std::pair<rope_ptr, rope_ptr> split(size_type pos, rope_ptr this_rope, Allocator alloc = Allocator())
		{
			if (is_leaf())
			{
				// Split leaf node
				if (pos == 0)
					return {nullptr, this_rope};

				if (pos >= weight_)
					return {this_rope, nullptr};

				auto left_view	= boost::core::basic_string_view<CharT>(blob_, pos);
				auto right_view = boost::core::basic_string_view<CharT>(blob_ + pos, weight_ - pos);

				auto* left_rope = new basic_rope<CharT, Traits, Allocator>(left_view);

				if (!left_rope)
					detail::throw_bad_alloc();

				auto* right_rope = new basic_rope<CharT, Traits, Allocator>(right_view);

				if (!right_rope)
				{
					delete left_rope;
					left_rope = nullptr;

					detail::throw_bad_alloc();
				}

				return {left_rope, right_rope};
			}

			// Internal node
			if (pos < weight_)
			{
				auto [ll, lr] = left_->impl_->split(pos, left_, alloc);
				return {ll, concat(lr, right_, alloc)};
			}
			else if (pos > weight_)
			{
				auto [rl, rr] = right_->impl_->split(pos - weight_, right_, alloc);
				return {concat(left_, rl, alloc), rr};
			}
			else
			{
				return {left_, right_};
			}
		}

		rope_ptr insert(size_type pos, const boost::core::basic_string_view<CharT>& str, rope_ptr this_rope, Allocator alloc = Allocator())
		{
			auto [left, right] = split(pos, this_rope, alloc);
			auto* middle	   = new basic_rope<CharT, Traits, Allocator>(str);
			return concat(concat(left, middle, alloc), right, alloc);
		}

		rope_ptr substr(size_type pos, size_type len, rope_ptr this_rope, Allocator alloc = Allocator())
		{
			auto [_, right] = split(pos, this_rope, alloc);

			if (!right)
				return nullptr;

			if (len == 0UL || len == basic_rope<CharT, Traits, Allocator>::npos)
				return right;

			auto [result, __] = right->impl_->split(len, right, alloc);

			return result;
		}

		bool starts_with(const boost::core::basic_string_view<CharT>& prefix) const
		{
			if (this->blob_ && std::string(this->blob_).starts_with(prefix))
				return true;

			return false;
		}

		bool ends_with(const boost::core::basic_string_view<CharT>& suffix) const
		{
			if (this->blob_ && std::string(this->blob_).ends_with(suffix))
				return true;

			return false;
		}

		bool equals(const tree_impl* other) const
		{
			if (!other)
				return false;
			if (total_size() != other->total_size())
				return false;

			// Compare character by character (could be optimized)
			size_type sz = total_size();
			for (size_type i = 0; i < sz; ++i)
			{
				if (at(i) != other->at(i))
					return false;
			}
			return true;
		}

		bool equals(const boost::core::basic_string_view<CharT>& str) const
		{
			if (total_size() != str.size())
				return false;

			for (size_type i = 0; i < str.size(); ++i)
			{
				if (at(i) != str[i])
					return false;
			}
			return true;
		}

	private:
		bool check_prefix(const boost::core::basic_string_view<CharT>& prefix, size_type& checked) const
		{
			if (is_leaf())
			{
				size_type to_check = std::min(weight_, prefix.size() - checked);

				if (Traits::compare(blob_, (prefix.data() + checked), to_check) != 0)
					return false;

				return true;
			}

			// Internal node
			if (left_ && !left_->impl_->check_prefix(prefix, checked))
				return false;
			if (right_)
				return right_->impl_->check_prefix(prefix, checked);

			return true;
		}

		bool check_suffix(const boost::core::basic_string_view<CharT>& suffix, size_type rope_pos, size_type suffix_pos) const
		{
			if (is_leaf())
			{
				if (rope_pos >= weight_)
					return suffix_pos >= suffix.size();

				size_type to_check = std::min(weight_ - rope_pos, suffix.size() - suffix_pos);

				if (Traits::compare(blob_ + rope_pos, (suffix.data() + suffix_pos), to_check) != 0)
					return false;

				return true;
			}

			if (left_ && !left_->impl_->check_suffix(suffix, rope_pos, suffix_pos))
				return false;
			if (right_)
				return right_->impl_->check_suffix(suffix, rope_pos, suffix_pos);

			return true;
		}
	};

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::~basic_rope()
	{
		delete impl_;
		impl_ = nullptr;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>&
	basic_rope<CharT, Traits, Allocator>::operator=(
		basic_rope<CharT, Traits, Allocator>&& other)
	{
		impl_ = std::exchange(other.impl_, nullptr);
		return *this;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::basic_rope(
		basic_rope<CharT, Traits, Allocator>&& other)
	{
		impl_ = std::exchange(other.impl_, nullptr);
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::basic_rope(
		const boost::core::basic_string_view<CharT>& in)
		: impl_(new tree_impl(in, Allocator()))
	{
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::begin()
	{
		if (impl_->is_leaf())
			return this;

		return impl_->left_ ? impl_->left_ : impl_->right_;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::end()
	{
		return nullptr;
	}

	/// \brief
	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::rbegin()
	{
		if (impl_->is_leaf())
			return this;

		auto last = impl_->right_;
		auto ret  = nullptr;

		while (last)
		{
			ret	 = last;
			last = last->right_;
		}

		return ret;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::rend()
	{
		return nullptr;
	}

	template <class CharT, class Traits, class Allocator>
	const basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::cbegin() const
	{
		if (impl_->is_leaf())
			return this;

		return impl_->left_ ? impl_->left_ : impl_->right_;
	}

	template <class CharT, class Traits, class Allocator>
	const basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::cend() const
	{
		return nullptr;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::size_type
	basic_rope<CharT, Traits, Allocator>::size() const
	{
		return impl_->size();
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::empty() const
	{
		return impl_->size() < 1UL;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>&
	basic_rope<CharT, Traits, Allocator>::substr(size_type pos, const size_type n)
	{
		if (!impl_)
			detail::throw_bad_alloc();

		auto* result = impl_->substr(pos, n, this, Allocator());

		if (!result)
			detail::throw_bad_alloc();

		return *result;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::size_type
	basic_rope<CharT, Traits, Allocator>::at(const boost::core::basic_string_view<CharT>& needle)
	{
		if (!impl_ || needle.empty())
			return npos;

		size_type rope_size = impl_->size();

		if (needle.size() > rope_size)
			return npos;

		for (size_type i{}; i <= (rope_size - needle.size()); ++i)
		{
			bool match{true};

			for (size_type j{}; j < needle.size(); ++j)
			{
				if (impl_->at(i + j) != needle[j])
				{
					match = false;
					break;
				}
			}

			if (match)
				return i;
		}

		return npos;
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::starts_with(const basic_rope<CharT>& other)
	{
		if (!impl_ || !other.impl_)
			return false;

		size_type other_size = other.impl_->size();

		if (other_size > impl_->size())
			return false;

		for (size_type i = 0; i < other_size; ++i)
		{
			if (impl_->at(i) != other.impl_->at(i))
				return false;
		}

		return true;
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::ends_with(const basic_rope<CharT>& other)
	{
		if (!impl_ || !other.impl_)
			return false;

		size_type this_size	 = impl_->size();
		size_type other_size = other.impl_->size();

		if (other_size > this_size)
			return false;

		size_type offset = this_size - other_size;

		for (size_type i = 0; i < other_size; ++i)
		{
			if (impl_->at(offset + i) != other.impl_->at(i))
				return false;
		}

		return true;
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::starts_with(const boost::core::basic_string_view<CharT>& prefix)
	{
		return impl_->starts_with(prefix);
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::ends_with(const boost::core::basic_string_view<CharT>& suffix)
	{
		return impl_->ends_with(suffix);
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::operator==(const basic_rope& other)
	{
		if (!impl_ && !other.impl_)
			return true;

		if (!impl_ || !other.impl_)
			return false;

		return impl_->equals(other.impl_);
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::operator!=(const basic_rope& other)
	{
		if (!impl_)
			return false;

		return !(*this == other);
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::operator==(const boost::core::basic_string_view<CharT>& str)
	{
		if (!impl_)
			return false;

		return impl_->equals(str);
	}

	template <class CharT, class Traits, class Allocator>
	std::basic_string<typename basic_rope<CharT, Traits, Allocator>::value_type>
	basic_rope<CharT, Traits, Allocator>::to_string()
	{
		if (!impl_)
			detail::throw_bad_alloc();

		std::basic_string<typename basic_rope<CharT, Traits, Allocator>::value_type> out{};

		size_t total{};

		for (auto p = impl_; p; p = p->right_ ? p->right_->impl_ : nullptr)
			total += p->capacity_;

		out.reserve(total);

		auto start = impl_;

		while (start)
		{
			out.append(start->blob_, start->capacity_);

			if (start->right_)
				start = start->right_->impl_;
			else
				break;
		}

		return std::move(out);
	}

	template <class CharT, class Traits, class Allocator>
	std::basic_string<typename basic_rope<CharT, Traits, Allocator>::value_type>
	basic_rope<CharT, Traits, Allocator>::to_string() const
	{
		return this->to_string();
	}

	template <class CharT, class Traits, class Allocator>
	bool basic_rope<CharT, Traits, Allocator>::operator!=(const boost::core::basic_string_view<CharT>& str)
	{
		return !(*this == str);
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::concat(rope_ptr right)
	{
		if (!impl_)
			detail::throw_bad_alloc();

		return impl_->concat(this, right);
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::insert(size_type									 pos,
																								const boost::core::basic_string_view<CharT>& text,
																								rope_ptr									 left) const
	{
		if (!impl_)
			detail::throw_bad_alloc();

		return impl_->insert(pos, text, left);
	}
	template <class CharT, class Traits, class Allocator>

	basic_rope<CharT, Traits, Allocator>::pointer basic_rope<CharT, Traits, Allocator>::operator*() const
	{
		return this->to_string();
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::operator++()
	{
        if (!impl_) return this;

		if (!impl_->right_)
			return impl_->left_;

		return impl_->right_ ? impl_->right_ : this;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::operator++(int n)
	{
		rope_ptr ret{};
		while (n)
		{
			ret = ret->operator++();
			--n;
		}

		return ret ? ret : this;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::operator--()
	{
		return impl_->left_;
	}

	template <class CharT, class Traits, class Allocator>
	basic_rope<CharT, Traits, Allocator>::rope_ptr basic_rope<CharT, Traits, Allocator>::operator--(int n)
	{
		rope_ptr ret{};
		while (n && ret)
		{
			ret = ret->operator--();
			--n;
		}

		return ret;
	}

} // namespace ocl::tproc

#endif
